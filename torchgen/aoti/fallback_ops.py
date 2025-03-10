# Be extra careful when you edit this file, because it affects AOTInductor ABI compatbility. See
# https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436
# for details.
#
# The inductor_fallback_ops list is based on the fallback ops from torch/_inductor/lowering.py.
# Generally speaking, it is ok to add a new op to the list, but you need to run
# `python torchgen/gen.py --update-aoti-c-shim` in order to regenerate C shim header files.
# But it is NOT ok to remove an existing fallback op from the list, since that will break
# some existing AOTInductor-compiled models.
inductor_fallback_ops = {
    "aten._adaptive_avg_pool2d_backward.default",
    "aten._adaptive_avg_pool2d.default",
    "aten._adaptive_avg_pool3d.default",
    "aten._adaptive_avg_pool3d_backward.default",
    "aten.adaptive_max_pool2d_backward.default",
    "aten.adaptive_max_pool2d.default",
    "aten.adaptive_max_pool3d.default",
    "aten.adaptive_max_pool3d_backward.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.addbmm.default",
    "aten._addmm_activation.default",
    "aten.addmm.out",
    "aten.addmv.default",
    "aten.angle.default",
    "aten.avg_pool2d_backward.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool3d_backward.default",
    "aten.avg_pool3d.default",
    "aten.baddbmm.out",
    "aten.bernoulli_.float",
    "aten.bernoulli_.Tensor",
    "aten.bmm.out",
    "aten.bucketize.Tensor",
    "aten.cat.default",
    "aten._cdist_backward.default",
    "aten._cdist_forward.default",
    "aten.cholesky_inverse.default",
    "aten.cholesky_solve.default",
    "aten.convolution_backward.default",
    "aten._cudnn_rnn.default",
    "aten.convolution.default",
    "aten.cummax.default",
    "aten.cummin.default",
    "aten.cumprod.default",
    "aten.cumsum.default",
    "aten._dyn_quant_matmul_4bit.default",
    "aten._dyn_quant_pack_4bit_weight.default",
    "aten._efficient_attention_backward.default",
    "aten._efficient_attention_forward.default",
    "aten._efficientzerotensor.default",
    "aten._embedding_bag.default",
    "aten._embedding_bag_dense_backward.default",
    "aten._embedding_bag_forward_only.default",
    "aten._embedding_bag_per_sample_weights_backward.default",
    "aten.exponential.default",
    "aten._fft_c2c.default",
    "aten._fft_r2c.default",
    "aten._flash_attention_backward.default",
    "aten._flash_attention_forward.default",
    "aten.fractional_max_pool2d_backward.default",
    "aten.fractional_max_pool2d.default",
    "aten.fractional_max_pool3d.default",
    "aten.fractional_max_pool3d_backward.default",
    "aten._fused_moving_avg_obs_fq_helper.default",
    "aten._fused_moving_avg_obs_fq_helper_functional.default",
    "aten.gcd.default",
    "aten.geqrf.default",
    "aten.grid_sampler_2d_backward.default",
    "aten.histc.default",
    "aten.histogram.bin_ct",
    "aten._histogramdd_from_bin_cts.default",
    "aten.index_put.default",
    "aten.index_reduce.default",
    "aten.index.Tensor",
    "aten._int_mm.out",
    "aten.kthvalue.default",
    "aten.logcumsumexp.default",
    "aten.lu_unpack.default",
    "aten.masked_select.default",
    "aten.masked_scatter.default",
    "aten.masked_scatter_backward.default",
    "aten.max_pool2d_with_indices_backward.default",
    "aten.max_pool2d_with_indices.default",
    "aten.max_pool3d_with_indices.default",
    "aten.max_pool3d_with_indices_backward.default",
    "aten.max_unpool2d.default",
    "aten.max_unpool3d.default",
    "aten.median.default",
    "aten.mm.out",
    "aten.mode.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.nanmedian.default",
    "aten.native_dropout.default",
    "aten.normal_functional.default",
    "aten.nonzero.default",
    "aten.ormqr.default",
    "aten._pdist_backward.default",
    "aten._pdist_forward.default",
    "aten.polar.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.rand.default",
    "aten.rand.generator",
    "aten.randint.default",
    "aten.randint.generator",
    "aten.randint.low",
    "aten.randint.low_out",
    "aten.randn.default",
    "aten.randn.generator",
    "aten.randperm.default",
    "aten.repeat_interleave.Tensor",
    "aten.replication_pad1d_backward.default",
    "aten.replication_pad2d_backward.default",
    "aten.reshape.default",
    "aten.resize_.default",
    "aten.resize_as_.default",
    "aten._scaled_dot_product_efficient_attention_backward.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_flash_attention_backward.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_cudnn_attention_backward.default",
    "aten._scaled_dot_product_cudnn_attention.default",
    "aten._scaled_dot_product_flash_attention_for_cpu_backward.default",
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_fused_attention_overrideable_backward.default",
    "aten._scaled_dot_product_fused_attention_overrideable.default",
    "aten._scaled_mm.default",
    "aten._scaled_mm.out",
    "aten.scatter_reduce.two_out",
    "aten.scatter.src_out",
    "aten.scatter.value_out",
    "aten.searchsorted.Scalar",
    "aten.searchsorted.Tensor",
    "aten._segment_reduce_backward.default",
    "aten.segment_reduce.default",
    "aten.set_.source_Tensor",
    "aten.slice.Tensor",
    "aten.soft_margin_loss_backward.default",
    "aten.sort.default",
    "aten.sort.stable",
    "aten._thnn_fused_lstm_cell.default",
    "aten.topk.default",
    "aten._to_sparse.default",
    "aten.to_sparse.default",
    "aten.triangular_solve.default",
    "aten._trilinear.default",
    "aten.uniform.default",
    "aten.upsample_bicubic2d_backward.default",
    "aten.upsample_linear1d_backward.default",
    "aten.upsample_trilinear3d_backward.default",
    "aten.view_as_complex.default",
    "aten.view_as_real.default",
    "aten.view.dtype",
    "aten._weight_int8pack_mm.default",
    "aten._weight_int4pack_mm_for_cpu.default",
}
