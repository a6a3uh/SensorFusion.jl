module SensorFusion

using LinearAlgebra
using ForwardDiff
using Transducers

include("Models.jl")
include("Kalman.jl")

end # module SensorFusion
