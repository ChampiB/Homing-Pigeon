/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_JIT_GEMM_CONVOLUTION_HPP
#define CPU_JIT_GEMM_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "gemm_convolution_utils.hpp"
#include "gemm/gemm.hpp"
#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct gemm_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                           forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                           this->desc()->src_desc.data_type,
                           this->desc()->weights_desc.data_type,
                           this->desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(), data_type::f32
                                   == this->desc()->bias_desc.data_type)
                && this->src_pd_.desc()->format == src_format()
                && this->dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format()
                && this->is_gemm_conv_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), weights_pd(0), dst_pd(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            auto const &po = this->attr()->post_ops_;
            auto is_eltwise = [&](int idx)
            { return po.entry_[idx].is_eltwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };

            switch (po.len_) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
            }
            return false;
        }
    };

    gemm_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true), eltwise_(nullptr)
    {
        const auto &post_ops = pd()->attr()->post_ops_;
        const data_t one = 1.0, zero = 0.0;
        beta_ = post_ops.find(primitive_kind::sum) >= 0 ? one : zero;

        const int entry_idx = post_ops.find(primitive_kind::eltwise);
        if (entry_idx != -1) eltwise_ = new ref_eltwise_scalar_fwd_t(
                post_ops.entry_[entry_idx].eltwise);
    }

    ~gemm_convolution_fwd_t() {
        delete eltwise_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    data_t beta_;

    ref_eltwise_scalar_fwd_t* eltwise_;
};

struct gemm_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && this->diff_src_pd_.desc()->format == src_format()
                && this->diff_dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_pd(), weights_pd(0), diff_dst_pd(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    gemm_convolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}
    ~gemm_convolution_bwd_data_t() {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct gemm_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
            && this->set_default_params() == status::success
            && this->desc()->prop_kind == backward_weights
            && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                       alg_kind::convolution_direct)
            && !this->has_zero_dim_memory()
            && utils::everyone_is(data_type::f32,
                    this->desc()->src_desc.data_type,
                    this->desc()->diff_weights_desc.data_type,
                    this->desc()->diff_dst_desc.data_type)
            && IMPLICATION(this->with_bias(),
                    data_type::f32 == this->desc()->diff_bias_desc.data_type)
            && this->src_pd_.desc()->format == src_format()
            && this->diff_dst_pd_.desc()->format == src_format()
            && this->diff_weights_pd_.desc()->format == wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), diff_weights_pd(0), diff_dst_pd(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(wei_format()));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    gemm_convolution_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}
    ~gemm_convolution_bwd_weights_t() {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
