// CSV and metrics-JSON writing.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace era {

// RFC 4180: quote fields containing a comma, quote or newline, doubling quotes.
// Player names come from the warehouse, so they cannot be assumed comma-free.
std::string csv_escape(const std::string& field);

// Split one record into fields, undoing csv_escape. A quoted field spanning a
// line break is not handled: no column this project writes contains a newline.
std::vector<std::string> csv_split_line(const std::string& line);

class CsvWriter {
   public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header);

    void write(const std::string& value);
    void write(double value);
    // Kept distinct from the double overload: a float32 latent widened to double
    // needs ~17 digits to round-trip, but only ~9 as a float.
    void write(float value);
    void write(int64_t value);
    void end_row();

   private:
    void separate();

    std::ofstream out_;
    bool at_row_start_ = true;
};

struct Metrics {
    int64_t rows = 0;
    double test_value_mse = 0.0;
    double test_value_r2 = 0.0;
    double test_era_accuracy = 0.0;
    double test_era_balanced_accuracy = 0.0;
    double fresh_lr_era_probe_balanced = 0.0;
    double random_era_accuracy = 0.0;
    std::vector<std::pair<std::string, double>> per_decade_era_accuracy;
    int64_t epochs_trained = 0;
};

void write_metrics_json(const std::string& path, const Metrics& metrics);

}  // namespace era
