#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;

class HofmeyrKdeOp : public OpKernel {
 public:
  explicit HofmeyrKdeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& x_tensor = context->input(0);
    const Tensor& y_tensor = context->input(1);
    const Tensor& x_eval_tensor = context->input(2);
    const Tensor& betas_tensor = context->input(3);
    const Tensor& h_tensor = context->input(4);

    auto x = x_tensor.flat<double>();
    auto y = y_tensor.flat<double>();
    auto x_eval = x_eval_tensor.flat<double>();
    auto betas = betas_tensor.flat<double>();
    auto h = h_tensor.scalar<double>()(0);

    // Create an output tensor
    Tensor* estimations_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_eval_tensor.shape(),
                                                     &estimations_tensor));

    auto estimations = estimations_tensor->flat<double>();

    /* setup parameters and object to be returned */
    int n = x.size();
    int n_eval = x_eval.size();
    int ord = betas.size() - 1;
    double denom;
    double exp_mult;

    /* Ly and Ry store the recursive sums used in fast kernel computations. See Fast Exact Evaluation of Univariate Kernel Sums (Hofmeyr, 2019)
    * for details
    */
    double Ly[ord+1][n];
    double Ry[ord+1][n];

    for(int i = 0; i <= ord; i++) Ly[i][0] = std::pow(-x(0), i) * y(0);
    for(int i = 1; i < n; i++){
        for(int j = 0; j <= ord; j++){
            Ly[j][i] = std::pow(-x(i), j) * y(i) + std::exp((x(i-1) - x(i)) / h) * Ly[j][i - 1];
            Ry[j][n - i - 1] = std::exp((x(n - i - 1) - x(n - i)) / h)*(std::pow(x(n - i), j) * y(n-i) + Ry[j][n - i]);
        }
    }


    int counts[n_eval];
    int count = 0;
    /* counts determined by looping, and since x and x_eval are sorted
    * counts(i+1) >= counts(i), meaning counting does not need to
    * restart for each
    */
    for(int i = 0; i < n_eval; i++){
        if(x_eval(i) >= x(n - 1)){
            counts[i] = n;
        }
        else{
            while(count < n && x(count) <= x_eval(i)){
                count += 1;
            }
            counts[i] = count;
        }
    }

    /* next loop over the terms in the polynomial in the kernel std::expression.
    * orddo represents the std::exponent, increasing to the order of the polynomial
    */
    for(int orddo = 0; orddo <= ord; orddo++){
        
        int coefs[orddo + 1]; /* coefs are the combinatorial coefficients in the binomial std::expansion */
        coefs[0] = coefs[orddo] = 1;
        if(orddo>1){
            double num = 1;
            for(int j = 2; j <= orddo; j++) num *= j;
            double denom1 = 1;
            double denom2 = num / orddo;
            for(int i = 2; i <= orddo; i++){
                coefs[i - 1] = num / denom1 / denom2;
                denom1 *= i;
                denom2 /= (orddo - i + 1);
            }
        }
        denom = std::pow(h, orddo); /* denominator for corresponding std::exponent is h^orddo */
        int ix; /* ix is the location of x_eval values in the x values, i.e., term in counts */

        /* next loop over evaluation points and compute the contribution to the kernel sum
        * from current polynomial term in the kernel
        */
        for(int i = 0; i < n_eval; i++){
            ix = round(counts[i]);
            if(ix == 0){
                exp_mult = std::exp((x_eval(i) - x(0)) / h);
                estimations(i) += betas(orddo) * std::pow(x(0) - x_eval(i), orddo) / denom * exp_mult * y(0);
                for(int j = 0; j <= orddo; j++) estimations(i) += betas(orddo) * coefs[j] * std::pow(-x_eval(i), orddo - j) * Ry[j][0] / denom * exp_mult;
            }
            else{
                exp_mult = std::exp((x(ix-1) - x_eval(i)) / h);
                for(int j = 0; j <= orddo; j++) estimations(i) += betas(orddo) * coefs[j] * (std::pow(x_eval(i), orddo - j) * Ly[j][ix - 1] * exp_mult + std::pow(-x_eval(i), orddo - j) * Ry[j][ix - 1] / exp_mult) / denom;
            }
        }
    }

    //Check for and prevent NaNs
    for(int i = 0; i < n_eval; i++){
        if(std::isnan(estimations(i))) {
            estimations(i) = 0.0;
        }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("HofmeyrKde").Device(DEVICE_CPU), HofmeyrKdeOp);

REGISTER_OP("HofmeyrKde")
    .Input("x: double")
    .Input("y: double")
    .Input("x_eval: double")
    .Input("h: double")
    .Input("betas: double")
    .Output("estimations: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });