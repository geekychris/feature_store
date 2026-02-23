package com.platform.featurestore.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

import java.time.Duration;

@Configuration
@ConditionalOnProperty(name = "feature-store.redis.enabled", havingValue = "true", matchIfMissing = true)
public class RedisConfig {

    @Value("${feature-store.redis.cache-ttl-seconds:3600}")
    private int cacheTtlSeconds;

    @Value("${feature-store.redis.key-prefix:fs:}")
    private String keyPrefix;

    @Bean
    public RedisTemplate<String, byte[]> featureStoreRedisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, byte[]> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(RedisSerializer.byteArray());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setHashValueSerializer(RedisSerializer.byteArray());
        template.afterPropertiesSet();
        return template;
    }

    @Bean
    public Duration redisCacheTtl() {
        return Duration.ofSeconds(cacheTtlSeconds);
    }

    @Bean
    public String redisKeyPrefix() {
        return keyPrefix;
    }
}
