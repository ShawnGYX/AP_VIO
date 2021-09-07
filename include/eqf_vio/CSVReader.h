#pragma once

#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "eigen3/Eigen/Core"
#include "liepp/SE3.h"
#include "liepp/SO3.h"
#include "liepp/SOT3.h"

class CSVLine {
  protected:
    friend class CSVReader;
    template <typename Derived> friend CSVLine& operator>>(CSVLine& line, Eigen::MatrixBase<Derived>& a);
    template <typename Derived> friend CSVLine& operator<<(CSVLine& line, const Eigen::MatrixBase<Derived>& a);
    template <typename Derived> friend CSVLine& operator>>(CSVLine& line, Eigen::QuaternionBase<Derived>& q);
    template <typename Derived> friend CSVLine& operator<<(CSVLine& line, const Eigen::QuaternionBase<Derived>& q);

    std::deque<std::string> data;

    void readLine(std::istream& lineStream, const char& delim = ',') {
        data.clear();
        std::string entry;
        while (std::getline(lineStream, entry, delim)) {
            data.emplace_back(entry);
        }
    }

  public:
    CSVLine() = default;
    CSVLine(std::istream& lineStream, const char& delim = ',') { readLine(lineStream, delim); }
    std::string operator[](const size_t& idx) const { return data[idx]; }
    size_t size() const { return data.size(); }

    // template <typename T> CSVLine(const T& d) { *this << d; }

    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true> CSVLine& operator>>(T& d) {
        std::stringstream(data.front()) >> d;
        data.pop_front();
        return *this;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true> CSVLine& operator<<(const T& d) {
        std::stringstream ss;
        ss << d;
        data.emplace_back(ss.str());
        return *this;
    }

    CSVLine& operator>>(std::string& s) {
        s = data.front();
        data.pop_front();
        return *this;
    }

    CSVLine& operator<<(const std::stringstream& ss) {
        data.emplace_back(ss.str());
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const CSVLine& line) {
    for (int i = 0; i < line.size() - 1; ++i) {
        os << line[i] << ", ";
    }
    os << line[line.size() - 1];
    return os;
}

template <typename Derived> CSVLine& operator>>(CSVLine& line, Eigen::MatrixBase<Derived>& a) {
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            line >> a(i, j);
        }
    }
    return line;
}

template <typename Derived> CSVLine& operator<<(CSVLine& line, const Eigen::MatrixBase<Derived>& a) {
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            line << a(i, j);
        }
    }
    return line;
}

template <typename Derived> CSVLine& operator>>(CSVLine& line, Eigen::QuaternionBase<Derived>& q) {
    return line >> q.w() >> q.x() >> q.y() >> q.z();
}

template <typename Derived> CSVLine& operator<<(CSVLine& line, const Eigen::QuaternionBase<Derived>& q) {
    return line << q.w() << q.x() << q.y() << q.z();
}

inline CSVLine& operator<<(CSVLine& line, const liepp::SO3d& R) { return line << R.asQuaternion(); }
inline CSVLine& operator>>(CSVLine& line, liepp::SO3d& R) {
    Eigen::Quaterniond q;
    line >> q;
    R.fromQuaternion(q);
    return line;
}

inline CSVLine& operator<<(CSVLine& line, const liepp::SE3d& P) { return line << P.x << P.R; }
inline CSVLine& operator>>(CSVLine& line, liepp::SE3d& P) { return line >> P.x >> P.R; }

inline CSVLine& operator<<(CSVLine& line, const liepp::SOT3d& Q) { return line << Q.a << Q.R; }
inline CSVLine& operator>>(CSVLine& line, liepp::SOT3d& Q) { return line >> Q.a >> Q.R; }

class CSVReader {
  protected:
    std::istream* csvPtr;
    CSVLine csvLine;
    char delim;

  public:
    CSVReader() { csvPtr = NULL; }
    CSVReader(std::istream& f, const char& delim = ',') : delim(delim) {
        csvPtr = f.good() ? &f : NULL;
        readNextLine();
    }

    CSVReader begin() { return *this; }
    CSVReader end() { return CSVReader(); }
    CSVLine operator*() { return csvLine; }
    bool operator==(const CSVReader& other) const {
        return (this->csvPtr == other.csvPtr) || ((this->csvPtr == NULL) && (other.csvPtr == NULL));
    }
    bool operator!=(const CSVReader& other) const { return !(*this == other); }

    CSVReader& operator++() {
        if (csvPtr) {
            // Try to read the next line
            if (!readNextLine()) {
                csvPtr = NULL;
            }
        }
        return *this;
    }

    bool readNextLine() {
        std::string lineString;
        bool fileNotEmpty = (bool)std::getline(*csvPtr, lineString, '\n');
        if (fileNotEmpty) {
            std::stringstream lineStream(lineString);
            csvLine.readLine(lineStream, delim);
        } else {
            csvLine = CSVLine();
        }
        return fileNotEmpty;
    }
};
