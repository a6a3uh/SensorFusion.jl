abstract type Model end
abstract type Linear <: Model end

"Linear system equation"
(m::Linear)(x::AbstractVector{<:Real}, u=0) =
    A(m) * x .+ B(m) * u

"State matrix"
A(::Linear, _...) = missing

"Control matrix"
B(::Linear, _...) = 0

Q(::Linear, _...) = missing

"Model size"
xsize(m::Linear) = size(A(m), 2)
esize(m::Linear) = size(Q(m), 2)

cov(m::Model, x, u=0) = Q(m, x, u) * Q(m, x, u)'

"Covariance propagation common formula for linear
and linearized systems"
(m::Model)(P::AbstractMatrix, x, u=0) =
    A(m, x, u) * P * A(m, x, u)' .+ cov(m, x, u) # broadcasting for scalar A * P * A' case

abstract type Nonlinear <: Model end

"Linearized state matrix"
A(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(x->m(x, u), x)

"Linearized control matrix"
B(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(x->m(x, u), u)

Q(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(e->m(x, u, e), zeros(esize(m)))

esize(m::Nonlinear) = missing
