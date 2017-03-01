// zhxfl 2016.02.28
#pragma once

#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <initializer_list>

using std::ostream;
using std::vector;
using std::initializer_list;

namespace tensor{

class Size {
private:
    const constexpr static int maxLen = 5;
    std::vector<int>_size;
public:
    explicit Size() {
        _size.reserve(maxLen);
    }

    template<typename T1, typename... T2, 
        typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
            explicit Size(T1 value, T2... args){
                _size.reserve(maxLen);
                _size.push_back(value);
                init(args...);
            }


    template<typename T1, typename... T2,
        typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
            void init(T1 value, T2... args){
                _size.push_back(value);
                init(args...);
            }
    void init(){};

    inline int& operator[](int idx){
        CHECK(idx <(int) _size.size());
        return _size[idx];
    }

    inline size_t size() const {
        return _size.size();
    }

    Size(const initializer_list<int> list)
        :_size(vector<int>(list)) {
        }

    inline bool operator == (Size& other) const {
        if(_size.size() != other.size())
            return false;
        for(size_t i = 0; i < _size.size(); i++){
            if(_size[i] != other[i])
                return false;
        }
        return true;
    }
    size_t count(){
        size_t count = 1;
        for(auto s : _size) count *=s;
        return count;
    }
};

class Stride {
    private:
        const constexpr static int maxLen = 5;
        std::vector<int>_stride;
    public:
        explicit Stride() {
            _stride.reserve(maxLen);
        }

        explicit Stride(Size& size) {
            _stride.reserve(maxLen);
            _stride.push_back(1);
            for(size_t i = 1; i < size.size(); i++){
                if(i == 1)
                    _stride.push_back(size[i - 1]);
                else{
                    _stride.push_back(size[i] * _stride[i - 1]);
                }
            }
        }

        template<typename T1, typename... T2, 
            typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
                explicit Stride(T1 value, T2...args) {
                    _stride.resize(maxLen);
                    _stride.push_back(value);
                    init(args...);
                }

        template<typename T1, typename... T2,
            typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
                void init(T1 value, T2... args){
                    _stride.push_back(value);
                    init(args...);
                }
        void init(){};

        Stride(const initializer_list<int> list)
            : _stride(vector<int>(list)) {
            }

        int &operator[](int idx){
            CHECK(idx < _stride.size());
            return _stride[idx];
        }

        inline size_t size() const {
            return _stride.size();
        }

        inline bool operator == (Stride& other) const {
            if(_stride.size() != other.size())
                return false;
            for(size_t i = 0; i < _stride.size(); i++){
                if(_stride[i] != other[i]){
                    return false;
                }
            }
            return true;
        }
};

class Offset {
    private:
        const constexpr static int maxLen = 5;
        std::vector<int> _offset;
    public:
        Offset(std::vector<int> offset){
            _offset = offset;
        }

        template<typename T1, typename... T2, 
            typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
                explicit Offset(T1 value, T2...args){
                    _offset.reserve(maxLen);
                    _offset.push_back(value);
                    init(args...);
                }

        template<typename T1, typename... T2,
            typename std::enable_if<std::is_integral<T1>::value, bool>::type = 0>
                void init(T1 value, T2... args){
                    _offset.push_back(value);
                    init(args...);
                }
        void init(){};

        Offset(const initializer_list<int> list)
            : _offset(vector<int>(list)) {
            }

        inline size_t size() const{
            return _offset.size();
        }

        int &operator[](int idx){
            CHECK(idx < _offset.size());
            return _offset[idx];
        }

        inline bool operator == (Offset& other) const {
            if(_offset.size() != other.size()){
                return false;
            }
            for(size_t i = 0; i < _offset.size(); i++){
                if(_offset[i] != other[i]){
                    return false; 
                }
            }
            return true;
        }
        friend Offset operator+ (Offset offset, const Offset& add);
        inline Offset& operator+= (Offset& add) {
            CHECK(_offset.size() == add.size());
            for(size_t i = 0; i < _offset.size(); i++){
                _offset[i] += add[i];
            }
            return *this;
        }
};

inline Offset operator+ (Offset offset, Offset& add) {
    CHECK(offset.size() == add.size());
    for(size_t i = 0; i < offset.size(); i++){
        offset[i] += add[i];
    }
    return offset;
}
}

