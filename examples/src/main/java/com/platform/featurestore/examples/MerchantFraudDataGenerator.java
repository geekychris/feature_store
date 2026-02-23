package com.platform.featurestore.examples;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * Generates a synthetic merchant fraud risk dataset (Java port of Python dataset.py).
 * <p>
 * Produces {@code n} merchants with 15 features and a binary {@code is_high_risk} label.
 * Feature distributions match the Python version: lognormal GMV, Poisson counts,
 * beta-distributed rates, etc. Positive rate ≈ 8%.
 */
public class MerchantFraudDataGenerator {

    // Feature schema — same as Python FEATURE_SCHEMA
    public static final List<FeatureSpec> FEATURE_SCHEMA = List.of(
            new FeatureSpec("gmv_30d",               "FLOAT64", "Gross merchandise volume last 30 days",      "DAILY",  86400),
            new FeatureSpec("gmv_90d",               "FLOAT64", "Gross merchandise volume last 90 days",      "DAILY",  86400),
            new FeatureSpec("txn_count_30d",         "FLOAT64", "Transaction count last 30 days",             "DAILY",  86400),
            new FeatureSpec("avg_txn_value",         "FLOAT64", "Average transaction value",                  "DAILY",  86400),
            new FeatureSpec("active_days_30d",       "FLOAT64", "Days with at least one transaction (30d)",   "DAILY",  86400),
            new FeatureSpec("chargeback_rate_90d",   "FLOAT64", "Chargeback rate over 90 days",              "DAILY",  86400),
            new FeatureSpec("refund_rate_30d",       "FLOAT64", "Refund rate over 30 days",                  "DAILY",  86400),
            new FeatureSpec("dispute_count_90d",     "FLOAT64", "Number of disputes in 90 days",             "DAILY",  86400),
            new FeatureSpec("fraud_reports_30d",     "FLOAT64", "Fraud reports received in 30 days",         "DAILY",  86400),
            new FeatureSpec("account_age_days",      "FLOAT64", "Days since merchant account creation",      "DAILY",  86400),
            new FeatureSpec("days_since_last_payout","FLOAT64", "Days since last payout settlement",         "HOURLY", 3600),
            new FeatureSpec("gmv_velocity_pct",      "FLOAT64", "GMV change rate (30d vs prior 30d)",        "DAILY",  86400),
            new FeatureSpec("txn_velocity_pct",      "FLOAT64", "Transaction count change rate",             "DAILY",  86400),
            new FeatureSpec("mcc_risk_score",        "FLOAT64", "Merchant category code risk (0-1)",         "WEEKLY", 604800),
            new FeatureSpec("country_risk_score",    "FLOAT64", "Country-level risk index (0-1)",            "WEEKLY", 604800)
    );

    public static final List<String> FEATURE_NAMES = FEATURE_SCHEMA.stream()
            .map(FeatureSpec::name).toList();

    public static final String LABEL_COL = "is_high_risk";
    public static final String VIEW_NAME = "merchant_fraud_gbdt_v1";
    public static final int VIEW_VERSION = 1;
    public static final String ENTITY_NAME = "merchant";

    public record FeatureSpec(String name, String dtype, String description,
                               String updateFrequency, int maxAgeSeconds) {}

    /** A single merchant row with features + label. */
    public record MerchantRow(String entityId, double[] features, int label) {
        public double feature(int i) { return features[i]; }
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /**
     * Generate a synthetic dataset of merchants.
     *
     * @param n    number of merchants (e.g. 50_000)
     * @param seed random seed for reproducibility
     */
    public static List<MerchantRow> generate(int n, long seed) {
        Random rng = new Random(seed);

        List<MerchantRow> rows = new ArrayList<>(n);

        // Pre-generate all feature arrays for vectorized-style computation
        double[] gmv30d = lognormal(rng, 10, 1.5, n);
        double[] gmv90d = new double[n];
        double[] txnCount30d = new double[n];
        double[] avgTxnValue = new double[n];
        double[] activeDays30d = new double[n];
        double[] accountAgeDays = new double[n];
        double[] daysSinceLastPayout = new double[n];
        double[] chargebackRate90d = new double[n];
        double[] refundRate30d = new double[n];
        double[] disputeCount90d = new double[n];
        double[] fraudReports30d = new double[n];
        double[] gmvVelocityPct = new double[n];
        double[] txnVelocityPct = new double[n];
        double[] mccRiskScore = new double[n];
        double[] countryRiskScore = new double[n];
        double[] riskScore = new double[n];

        for (int i = 0; i < n; i++) {
            gmv90d[i] = gmv30d[i] * (2.5 + rng.nextDouble());
            txnCount30d[i] = poisson(rng, 200);
            avgTxnValue[i] = gmv30d[i] / Math.max(txnCount30d[i], 1);
            activeDays30d[i] = 3 + rng.nextInt(28);
            accountAgeDays[i] = 7 + rng.nextInt(1818);
            daysSinceLastPayout[i] = rng.nextInt(30);

            double isNew = accountAgeDays[i] < 90 ? 1.0 : 0.0;
            chargebackRate90d[i] = clamp(betaSample(rng, 1 + isNew * 2, 50), 0, 0.2);
            refundRate30d[i] = clamp(betaSample(rng, 2, 30), 0, 0.3);
            disputeCount90d[i] = poisson(rng, 1.5 + isNew * 3);
            fraudReports30d[i] = poisson(rng, 0.3 + isNew * 1.5);

            gmvVelocityPct[i] = rng.nextGaussian() * 0.30 + 0.05;
            txnVelocityPct[i] = rng.nextGaussian() * 0.25 + 0.03;
            mccRiskScore[i] = rng.nextDouble();
            countryRiskScore[i] = rng.nextDouble();

            riskScore[i] = chargebackRate90d[i] * 5.0
                    + fraudReports30d[i] * 1.2
                    + disputeCount90d[i] * 0.3
                    + (1.0 / (accountAgeDays[i] + 1)) * 200
                    + Math.max(gmvVelocityPct[i], 0) * 0.8
                    + mccRiskScore[i] * 0.5
                    + countryRiskScore[i] * 0.3
                    - Math.log1p(gmv90d[i]) * 0.03
                    + rng.nextGaussian() * 0.15;
        }

        // Compute threshold at 92nd percentile
        double[] sorted = Arrays.copyOf(riskScore, n);
        Arrays.sort(sorted);
        double threshold = sorted[(int) (n * 0.92)];

        for (int i = 0; i < n; i++) {
            String entityId = String.format("m_%06d", i);
            double[] features = {
                    gmv30d[i], gmv90d[i], txnCount30d[i], avgTxnValue[i],
                    activeDays30d[i], chargebackRate90d[i], refundRate30d[i],
                    disputeCount90d[i], fraudReports30d[i], accountAgeDays[i],
                    daysSinceLastPayout[i], gmvVelocityPct[i], txnVelocityPct[i],
                    mccRiskScore[i], countryRiskScore[i]
            };
            int label = riskScore[i] > threshold ? 1 : 0;
            rows.add(new MerchantRow(entityId, features, label));
        }

        return rows;
    }

    /**
     * Compute the schema hash (same algorithm as Java server + Python).
     */
    public static int computeSchemaHash(List<String> featureNames) {
        try {
            String key = String.join(",", featureNames);
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(key.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format("%02x", b));
            return (int) (Long.parseLong(hex.substring(0, 8), 16) % Integer.MAX_VALUE);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 not available", e);
        }
    }

    /** Split dataset into train/test (stratified by label). */
    public static List<List<MerchantRow>> trainTestSplit(List<MerchantRow> data,
                                                          double testRatio, long seed) {
        List<MerchantRow> pos = data.stream().filter(r -> r.label() == 1).toList();
        List<MerchantRow> neg = data.stream().filter(r -> r.label() == 0).toList();

        List<MerchantRow> posShuffled = new ArrayList<>(pos);
        List<MerchantRow> negShuffled = new ArrayList<>(neg);
        Collections.shuffle(posShuffled, new Random(seed));
        Collections.shuffle(negShuffled, new Random(seed));

        int posTestCount = (int) (pos.size() * testRatio);
        int negTestCount = (int) (neg.size() * testRatio);

        List<MerchantRow> test = new ArrayList<>();
        test.addAll(posShuffled.subList(0, posTestCount));
        test.addAll(negShuffled.subList(0, negTestCount));

        List<MerchantRow> train = new ArrayList<>();
        train.addAll(posShuffled.subList(posTestCount, pos.size()));
        train.addAll(negShuffled.subList(negTestCount, neg.size()));

        Collections.shuffle(train, new Random(seed + 1));
        Collections.shuffle(test, new Random(seed + 1));

        return List.of(train, test);
    }

    // -----------------------------------------------------------------------
    // Distribution helpers
    // -----------------------------------------------------------------------

    private static double[] lognormal(Random rng, double mu, double sigma, int n) {
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = Math.exp(mu + sigma * rng.nextGaussian());
        }
        return result;
    }

    private static int poisson(Random rng, double lambda) {
        double L = Math.exp(-lambda);
        double p = 1.0;
        int k = 0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > L);
        return k - 1;
    }

    private static double betaSample(Random rng, double alpha, double beta) {
        double x = gammaSample(rng, alpha);
        double y = gammaSample(rng, beta);
        return x / (x + y);
    }

    private static double gammaSample(Random rng, double shape) {
        // Marsaglia and Tsang's method for shape >= 1
        if (shape < 1) {
            return gammaSample(rng, shape + 1) * Math.pow(rng.nextDouble(), 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x, v;
            do {
                x = rng.nextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = rng.nextDouble();
            if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }

    private static double clamp(double value, double min, double max) {
        return Math.min(Math.max(value, min), max);
    }
}
