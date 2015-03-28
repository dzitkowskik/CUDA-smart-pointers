/*
 * (C) Copyright Karol Dzitkowski 2015
 *
 */

#define CUDA_CALL(x) do { assert((x)==cudaSuccess); } while(0)

#ifndef CUDA_SMART_PTR_SHARED_PTR_HPP_
#define CUDA_SMART_PTR_SHARED_PTR_HPP_

#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>            // for std::swap
#include <atomic>
#include <cstddef>

template <class T> class cuda_shared_ptr
{
private:
    T * ptr;
    std::atomic<int> * cnt;

public:
    explicit cuda_shared_ptr()
    {
        init();
        CUDA_CALL( cudaMalloc((void**)&ptr, sizeof(T)) );
    }
    explicit cuda_shared_ptr(size_t size)
    {
        init();
        if (size == 0) ptr = nullptr;
        else CUDA_CALL( cudaMalloc((void**)&ptr, size) );
    }
    explicit cuda_shared_ptr(T * p = 0) : ptr(p) { init(); }

    ~cuda_shared_ptr()
    {
        if (decCounter())
        {
            CUDA_CALL( cudaFree(ptr) );
            delete cnt;
        }
    }
    void reset(T * p = 0)
    {
        assert(p == 0 || p != ptr);
        cuda_shared_ptr<T>(p).swap(*this);
    }
    T & operator*() const
    {
        assert(ptr != 0);
        return *ptr;
    }
    T * operator->() const
    {
        assert(ptr != 0);
        return ptr;
    }
    T * get() const { return ptr; }
    operator bool() const
    {
        return ptr != 0;
    }
    void swap(cuda_shared_ptr & other) noexcept
    {
        std::swap(ptr, other.ptr);
        std::swap(cnt, other.cnt);
    }
    int use_count()
    {
        return *cnt;
    }
private:
    bool decCounter(){ return --(*cnt) == 0; }
    void incCounter(){ (*cnt)++; }
    void init(){ cnt = new std::atomic<int>(); cnt->store(1); }

public:
    cuda_shared_ptr(cuda_shared_ptr const & r) : ptr(r.ptr), cnt(r.cnt)
    {
        incCounter();
    }

    /* NOT YET SUPPORTED
    cuda_shared_ptr( cuda_shared_ptr && r ) noexcept : ptr( r.ptr ), cnt()
    {
        swap(cnt, r.cnt);
        r.px = 0;
    }
    */

    cuda_shared_ptr & operator=(cuda_shared_ptr const & r) noexcept
    {
        cuda_shared_ptr<T>(r).swap(*this);
        return *this;
    }

    template<class Y>
    cuda_shared_ptr & operator=(cuda_shared_ptr<Y> const & r) noexcept
    {
        cuda_shared_ptr<T>(r).swap(*this);
        return *this;
    }
};

template<class T, class U> inline bool operator==(
    cuda_shared_ptr<T> const & a,
    cuda_shared_ptr<U> const & b) noexcept
{
    return a.get() == b.get();
}

template<class T, class U> inline bool operator!=(
    cuda_shared_ptr<T> const & a,
    cuda_shared_ptr<U> const & b) noexcept
{
    return a.get() != b.get();
}

template<class T> inline bool operator==(
    cuda_shared_ptr<T> const & p,
    std::nullptr_t ) noexcept
{
    return p.get() == 0;
}

template<class T> inline bool operator==(
    std::nullptr_t,
    cuda_shared_ptr<T> const & p) noexcept
{
    return p.get() == 0;
}

template<class T> inline bool operator!=(
    cuda_shared_ptr<T> const & p,
    std::nullptr_t) noexcept
{
    return p.get() != 0;
}

template<class T> inline bool operator!=(
    std::nullptr_t,
    cuda_shared_ptr<T> const & p) noexcept
{
    return p.get() != 0;
}

#endif
