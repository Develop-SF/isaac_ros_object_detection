/*
 * Filename: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src/detection2_d_array_vlm_filter.cpp
 * Path: /home/shinfang-ovx/workspaces/wy/isaac_ros_ws/src/isaac_ros_object_detection/isaac_ros_yolov8/src
 * Created Date: Monday, November 3rd 2025, 11:03:24 am
 * Author: Wen-Yu Chien
 * Description: Isaac ROS VLM BBOX Selector
 * Copyright (c) 2025 Copyright (c) 2025 Shinfang Global
 */
#pragma once

#include <cctype>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

inline std::string json_escape(const std::string& input) {
    std::ostringstream escaped;
    for (char ch : input) {
        switch (ch) {
            case '\"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    escaped << "\\u"
                            << std::hex << std::uppercase << std::setfill('0')
                            << std::setw(4) << static_cast<int>(static_cast<unsigned char>(ch))
                            << std::dec << std::nouppercase;
                } else {
                    escaped << ch;
                }
        }
    }
    return escaped.str();
}

inline std::string base64_encode(const std::vector<unsigned char>& data) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        unsigned int n = (static_cast<unsigned int>(data[i]) << 16) |
                         (static_cast<unsigned int>(data[i + 1]) << 8) |
                         static_cast<unsigned int>(data[i + 2]);
        encoded.push_back(table[(n >> 18) & 63]);
        encoded.push_back(table[(n >> 12) & 63]);
        encoded.push_back(table[(n >> 6) & 63]);
        encoded.push_back(table[n & 63]);
        i += 3;
    }

    if (i < data.size()) {
        unsigned int n = static_cast<unsigned int>(data[i]) << 16;
        encoded.push_back(table[(n >> 18) & 63]);
        if (i + 1 < data.size()) {
            n |= static_cast<unsigned int>(data[i + 1]) << 8;
            encoded.push_back(table[(n >> 12) & 63]);
            encoded.push_back(table[(n >> 6) & 63]);
            encoded.push_back('=');
        } else {
            encoded.push_back(table[(n >> 12) & 63]);
            encoded.push_back('=');
            encoded.push_back('=');
        }
    }

    return encoded;
}

inline std::optional<std::vector<unsigned char>> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        return std::nullopt;
    }
    return buffer;
}

inline std::string decode_json_string(const std::string& value) {
    std::string decoded;
    decoded.reserve(value.size());

    for (size_t i = 0; i < value.size(); ++i) {
        char ch = value[i];
        if (ch != '\\') {
            decoded.push_back(ch);
            continue;
        }

        if (i + 1 >= value.size()) {
            break;
        }

        char next = value[++i];
        switch (next) {
            case '\"': decoded.push_back('\"'); break;
            case '\\': decoded.push_back('\\'); break;
            case '/': decoded.push_back('/'); break;
            case 'b': decoded.push_back('\b'); break;
            case 'f': decoded.push_back('\f'); break;
            case 'n': decoded.push_back('\n'); break;
            case 'r': decoded.push_back('\r'); break;
            case 't': decoded.push_back('\t'); break;
            case 'u':
                if (i + 4 < value.size()) {
                    std::string hex = value.substr(i + 1, 4);
                    i += 4;
                    try {
                        char16_t code = static_cast<char16_t>(std::stoi(hex, nullptr, 16));
                        if (code <= 0x7F) {
                            decoded.push_back(static_cast<char>(code));
                        }
                    } catch (const std::exception&) {
                        // Ignore malformed unicode escapes and skip them.
                    }
                }
                break;
            default:
                decoded.push_back(next);
                break;
        }
    }

    return decoded;
}

inline std::string extract_output_text(const std::string& body) {
    std::istringstream stream(body);
    std::string line;
    std::string accumulated;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const std::string message_pattern = "\"message\":";
        size_t message_pos = line.find(message_pattern);
        if (message_pos == std::string::npos) {
            continue;
        }

        const std::string content_pattern = "\"content\":\"";
        size_t content_pos = line.find(content_pattern, message_pos);
        if (content_pos == std::string::npos) {
            continue;
        }
        content_pos += content_pattern.size();

        std::string raw;
        raw.reserve(line.size() - content_pos);

        bool escape = false;
        for (size_t i = content_pos; i < line.size(); ++i) {
            char ch = line[i];
            if (escape) {
                raw.push_back('\\');
                raw.push_back(ch);
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (ch == '\"') {
                content_pos = i + 1;
                break;
            }
            raw.push_back(ch);
        }

        accumulated += decode_json_string(raw);
    }

    return accumulated;
}

struct ResponseStatistics {
    long long input_tokens = 0;
    long long output_tokens = 0;
    double latency_ms = 0.0;
    double tokens_per_second = 0.0;
};

inline std::string extract_json_value(const std::string& source, const std::string& key) {
    const std::string pattern = "\"" + key + "\"";
    size_t pos = source.find(pattern);
    if (pos == std::string::npos) {
        return "";
    }
    pos = source.find(':', pos + pattern.size());
    if (pos == std::string::npos) {
        return "";
    }
    ++pos;
    while (pos < source.size() && std::isspace(static_cast<unsigned char>(source[pos]))) {
        ++pos;
    }
    if (pos >= source.size()) {
        return "";
    }
    if (source[pos] == '\"') {
        ++pos;
        size_t end = source.find('\"', pos);
        if (end == std::string::npos) {
            return "";
        }
        return source.substr(pos, end - pos);
    }
    size_t end = pos;
    while (end < source.size() &&
           (std::isdigit(static_cast<unsigned char>(source[end])) || source[end] == '.' ||
            source[end] == '-' || source[end] == '+' || source[end] == 'e' || source[end] == 'E')) {
        ++end;
    }
    return source.substr(pos, end - pos);
}

inline std::optional<long long> parse_long_long(const std::string& value) {
    if (value.empty()) {
        return std::nullopt;
    }
    try {
        size_t idx = 0;
        long long parsed = std::stoll(value, &idx);
        if (idx != value.size()) {
            return std::nullopt;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

inline ResponseStatistics compute_statistics(const std::string& source) {
    ResponseStatistics stats;

    const auto prompt_eval_count = parse_long_long(extract_json_value(source, "prompt_eval_count"));
    const auto eval_count = parse_long_long(extract_json_value(source, "eval_count"));
    const auto total_duration_ns = parse_long_long(extract_json_value(source, "total_duration"));
    const auto eval_duration_ns = parse_long_long(extract_json_value(source, "eval_duration"));

    if (prompt_eval_count) {
        stats.input_tokens = *prompt_eval_count;
    }
    if (eval_count) {
        stats.output_tokens = *eval_count;
    }
    if (total_duration_ns) {
        stats.latency_ms = static_cast<double>(*total_duration_ns) / 1'000'000.0;
    }
    if (eval_duration_ns && eval_count && *eval_duration_ns > 0) {
        double seconds = static_cast<double>(*eval_duration_ns) / 1'000'000'000.0;
        if (seconds > 0.0) {
            stats.tokens_per_second = static_cast<double>(*eval_count) / seconds;
        }
    }

    return stats;
}

inline bool has_statistics(const ResponseStatistics& stats) {
    return stats.input_tokens > 0 || stats.output_tokens > 0 ||
           stats.latency_ms > 0.0 || stats.tokens_per_second > 0.0;
}

inline void print_statistics(const ResponseStatistics& stats, std::ostream& out = std::cout) {
    if (!has_statistics(stats)) {
        return;
    }

    out << "=== Response Statistics ===" << std::endl;
    if (stats.input_tokens > 0) {
        out << "Input Tokens: " << stats.input_tokens << std::endl;
    }
    if (stats.output_tokens > 0) {
        out << "Output Tokens: " << stats.output_tokens << std::endl;
    }
    if (stats.latency_ms > 0.0) {
        std::ostringstream latency_stream;
        latency_stream << std::fixed << std::setprecision(2) << stats.latency_ms;
        out << "Latency (ms): " << latency_stream.str() << std::endl;
    }
    if (stats.tokens_per_second > 0.0) {
        std::ostringstream tps_stream;
        tps_stream << std::fixed << std::setprecision(2) << stats.tokens_per_second;
        out << "Tokens per Second (TPS): " << tps_stream.str() << std::endl;
    }
}
