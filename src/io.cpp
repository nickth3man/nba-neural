#include "io.hpp"

#include <array>
#include <charconv>
#include <stdexcept>

namespace era {
namespace {

// Shortest representation that round-trips, matching what Python's repr - and
// therefore pandas' to_csv and json.dump - writes. Fixed precision would render
// a rounded 2.523 as 2.5230000000000001.
template <typename T>
std::string format_number(T value) {
    std::array<char, 32> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (error != std::errc()) {
        throw std::runtime_error("cannot format number");
    }
    return std::string(buffer.data(), end);
}

std::string format_double(double value) { return format_number(value); }

}  // namespace

std::string csv_escape(const std::string& field) {
    if (field.find_first_of(",\"\n\r") == std::string::npos) {
        return field;
    }
    std::string escaped = "\"";
    for (const char c : field) {
        if (c == '"') {
            escaped += '"';
        }
        escaped += c;
    }
    escaped += '"';
    return escaped;
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header) {
    out_.open(path, std::ios::binary);
    if (!out_) {
        throw std::runtime_error("cannot write " + path);
    }
    for (const std::string& name : header) {
        write(name);
    }
    end_row();
}

void CsvWriter::separate() {
    if (!at_row_start_) {
        out_ << ',';
    }
    at_row_start_ = false;
}

void CsvWriter::write(const std::string& value) {
    separate();
    out_ << csv_escape(value);
}

void CsvWriter::write(double value) {
    separate();
    out_ << format_double(value);
}

void CsvWriter::write(float value) {
    separate();
    out_ << format_number(value);
}

void CsvWriter::write(int64_t value) {
    separate();
    out_ << value;
}

void CsvWriter::end_row() {
    out_ << '\n';
    at_row_start_ = true;
}

void write_metrics_json(const std::string& path, const Metrics& metrics) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << "{\n";
    out << "  \"rows\": " << metrics.rows << ",\n";
    out << "  \"test_value_mse\": " << format_double(metrics.test_value_mse) << ",\n";
    out << "  \"test_value_r2\": " << format_double(metrics.test_value_r2) << ",\n";
    out << "  \"test_era_accuracy\": " << format_double(metrics.test_era_accuracy) << ",\n";
    out << "  \"test_era_balanced_accuracy\": "
        << format_double(metrics.test_era_balanced_accuracy) << ",\n";
    out << "  \"fresh_lr_era_probe_balanced\": "
        << format_double(metrics.fresh_lr_era_probe_balanced) << ",\n";
    out << "  \"random_era_accuracy\": " << format_double(metrics.random_era_accuracy) << ",\n";
    out << "  \"per_decade_era_accuracy\": {\n";
    for (std::size_t i = 0; i < metrics.per_decade_era_accuracy.size(); ++i) {
        const auto& [decade, accuracy] = metrics.per_decade_era_accuracy[i];
        out << "    \"" << decade << "\": " << format_double(accuracy);
        out << (i + 1 < metrics.per_decade_era_accuracy.size() ? ",\n" : "\n");
    }
    out << "  },\n";
    out << "  \"epochs_trained\": " << metrics.epochs_trained << "\n";
    out << "}\n";
}

}  // namespace era
