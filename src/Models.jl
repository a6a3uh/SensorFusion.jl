abstract type Model end
abstract type Linear <: Model end

"Linear system equation"
(m::Linear)(x::AbstractVector{<:Real}, u=0, _...) =
    A(m) * x .+ B(m) * u

"State matrix"
A(::Linear) = missing

"Control matrix"
B(::Linear) = 0

Q(::Linear) = missing

"Model size"
xsize(m::Linear) = size(A(m), 1)
esize(m::Linear) = missing

cov(m::Model, x, u) = Q(m, x, u) * Q(m, x, u)'

"Covariance propagation common formula for linear
and linearized systems"
(m::Model)(P::AbstractMatrix, x, u=0) =
    A(m, x, u) * P * A(m, x, u)' + cov(m, x, u)

abstract type Nonlinear <: Model end

"Linearized state matrix"
A(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(x->m(x, u), x)

"Linearized control matrix"
B(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(x->m(x, u), u)

Q(m::Nonlinear, x, u=0) =
    ForwardDiff.jacobian(e->m(x, u, e), zeros(esize(m)))
