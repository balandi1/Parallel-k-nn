

// Wrapper around a pointer, for reading values from byte sequence.
class Reader {
    public:
        Reader(const char *p) : ptr{p} {}
        template <typename T>
        Reader &operator>>(T &o) {
            assert(uintptr_t(ptr)%sizeof(T) == 0);
            o = *(T *) ptr;
            ptr += sizeof(T);
            return *this;
        }
    private:
        const char *ptr;
};

// function to perform file operations
void readInputs(const std::string &fn);
void writeResult();
uint64_t generateResultID();