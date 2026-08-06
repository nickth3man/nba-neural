// Unit tests for the training loop helpers.
#include <gtest/gtest.h>

#include "config.hpp"
#include "train.hpp"

TEST(Train, LambdaScheduleRamps) {
    const double first = era::lambda_at(0, era::MAX_EPOCHS);
    const double middle = era::lambda_at(era::MAX_EPOCHS / 2, era::MAX_EPOCHS);
    const double last = era::lambda_at(era::MAX_EPOCHS - 1, era::MAX_EPOCHS);

    EXPECT_DOUBLE_EQ(first, 0.0);
    EXPECT_GT(middle, 0.0);
    EXPECT_LT(middle, era::LAMBDA_MAX);
    EXPECT_GT(last, 0.99 * era::LAMBDA_MAX);
    EXPECT_LT(first, middle);
    EXPECT_LT(middle, last);
}
