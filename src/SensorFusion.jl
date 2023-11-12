module SensorFusion

using LinearAlgebra
using ForwardDiff
using Transducers

include("Models.jl")
export Model, Linear, Nonlinear
export xsize, esize, A, B, Q, cov
include("Kalman.jl")
export Kalman, LinearKalman
export specifics, update, estimate

end # module SensorFusion
