#include "specdraft/model_file.hpp"

#include <cstring>
#include <stdexcept>

#include "specdraft/quant.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace specdraft {
namespace {

// Must match python/specdraft/export.py.
constexpr char MAGIC[4] = {'S', 'D', 'M', '2'};
constexpr uint32_t VERSION = 2;
constexpr size_t HEADER_BYTES = 128;
constexpr size_t ENTRY_BYTES = 72;
constexpr size_t NAME_BYTES = 40;

template <typename T>
T read_at(const uint8_t* base, size_t offset) {
    T value;
    std::memcpy(&value, base + offset, sizeof(T));  // the file is packed, so never cast
    return value;
}

}  // namespace

const char* format_name(Format format) {
    switch (format) {
        case Format::fp32:
            return "fp32";
        case Format::q4:
            return "q4";
        case Format::q8:
            return "q8";
    }
    return "?";
}

size_t row_bytes(Format format, uint32_t cols) {
    switch (format) {
        case Format::fp32:
            return static_cast<size_t>(cols) * sizeof(float);
        case Format::q4:
            return static_cast<size_t>(cols) / QK * sizeof(BlockQ4);
        case Format::q8:
            return static_cast<size_t>(cols) / QK * sizeof(BlockQ8);
    }
    return 0;
}

ModelFile::~ModelFile() {
    close();
}

ModelFile::ModelFile(ModelFile&& other) noexcept {
    *this = std::move(other);
}

ModelFile& ModelFile::operator=(ModelFile&& other) noexcept {
    if (this != &other) {
        close();
        mapping_ = other.mapping_;
        size_ = other.size_;
#if defined(_WIN32)
        file_handle_ = other.file_handle_;
        map_handle_ = other.map_handle_;
        other.file_handle_ = nullptr;
        other.map_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        config_ = other.config_;
        tensors_ = std::move(other.tensors_);
        other.mapping_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void ModelFile::close() {
#if defined(_WIN32)
    if (mapping_ != nullptr) {
        UnmapViewOfFile(mapping_);
    }
    if (map_handle_ != nullptr) {
        CloseHandle(map_handle_);
    }
    if (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
    map_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    if (mapping_ != nullptr) {
        munmap(mapping_, size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = -1;
#endif
    mapping_ = nullptr;
    size_ = 0;
    tensors_.clear();
}

ModelFile ModelFile::open(const std::string& path) {
    ModelFile file;

#if defined(_WIN32)
    file.file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file.file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot open " + path);
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file.file_handle_, &size)) {
        throw std::runtime_error("cannot size " + path);
    }
    file.size_ = static_cast<size_t>(size.QuadPart);
    file.map_handle_ = CreateFileMappingA(file.file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (file.map_handle_ == nullptr) {
        throw std::runtime_error("cannot map " + path);
    }
    file.mapping_ = MapViewOfFile(file.map_handle_, FILE_MAP_READ, 0, 0, 0);
#else
    file.fd_ = ::open(path.c_str(), O_RDONLY);
    if (file.fd_ < 0) {
        throw std::runtime_error("cannot open " + path);
    }
    struct stat info {};
    if (fstat(file.fd_, &info) != 0) {
        throw std::runtime_error("cannot size " + path);
    }
    file.size_ = static_cast<size_t>(info.st_size);
    file.mapping_ = mmap(nullptr, file.size_, PROT_READ, MAP_PRIVATE, file.fd_, 0);
    if (file.mapping_ == MAP_FAILED) {
        file.mapping_ = nullptr;
    }
#endif
    if (file.mapping_ == nullptr) {
        throw std::runtime_error("cannot map " + path);
    }
    if (file.size_ < HEADER_BYTES) {
        throw std::runtime_error(path + " is too small to be a model file");
    }

    const auto* base = static_cast<const uint8_t*>(file.mapping_);
    if (std::memcmp(base, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::runtime_error(path + " is not a specdraft model file");
    }
    const uint32_t version = read_at<uint32_t>(base, 4);
    if (version != VERSION) {
        throw std::runtime_error("unsupported model file version " + std::to_string(version));
    }
    const uint32_t count = read_at<uint32_t>(base, 8);
    const uint32_t data_start = read_at<uint32_t>(base, 12);

    ModelConfig& config = file.config_;
    config.vocab_size = read_at<uint32_t>(base, 16);
    config.hidden_size = read_at<uint32_t>(base, 20);
    config.intermediate_size = read_at<uint32_t>(base, 24);
    config.num_hidden_layers = read_at<uint32_t>(base, 28);
    config.num_attention_heads = read_at<uint32_t>(base, 32);
    config.num_key_value_heads = read_at<uint32_t>(base, 36);
    config.head_dim = read_at<uint32_t>(base, 40);
    config.vocab_limit = read_at<uint32_t>(base, 44);
    config.tie_word_embeddings = read_at<uint32_t>(base, 48);
    config.rms_norm_eps = read_at<double>(base, 52);
    config.rope_theta = read_at<double>(base, 60);

    if (HEADER_BYTES + static_cast<size_t>(count) * ENTRY_BYTES > file.size_ ||
        data_start > file.size_) {
        throw std::runtime_error(path + " has a truncated tensor directory");
    }

    for (uint32_t i = 0; i < count; ++i) {
        const size_t at = HEADER_BYTES + static_cast<size_t>(i) * ENTRY_BYTES;
        char name[NAME_BYTES + 1] = {};
        std::memcpy(name, base + at, NAME_BYTES);

        Tensor tensor;
        tensor.format = static_cast<Format>(read_at<uint32_t>(base, at + NAME_BYTES));
        const uint32_t ndim = read_at<uint32_t>(base, at + NAME_BYTES + 4);
        tensor.rows = read_at<uint32_t>(base, at + NAME_BYTES + 8);
        tensor.cols = (ndim == 1) ? 1 : read_at<uint32_t>(base, at + NAME_BYTES + 12);
        const uint64_t offset = read_at<uint64_t>(base, at + NAME_BYTES + 16);
        tensor.nbytes = static_cast<size_t>(read_at<uint64_t>(base, at + NAME_BYTES + 24));
        if (offset + tensor.nbytes > file.size_) {
            throw std::runtime_error(std::string("tensor ") + name + " runs past the end of " + path);
        }
        tensor.data = base + offset;
        file.tensors_.emplace(name, tensor);
    }
    return file;
}

const Tensor& ModelFile::tensor(const std::string& name) const {
    auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error("model file has no tensor named " + name);
    }
    return found->second;
}

std::vector<std::string> ModelFile::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        out.push_back(name);
    }
    return out;
}

}  // namespace specdraft
