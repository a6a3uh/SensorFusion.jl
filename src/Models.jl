export Model, Linear, Nonlinear
export xsize, esize, usize, ysize
export A, B, Q, cov

abstract type Model end
abstract type Linear <: Model end


"Linear system equation"
(m::Linear)(;
            x=zeros(xsize(m)),
            u=zeros(usize(m)),
            y=zeros(ysize(m)),
            e=zeros(esize(m))) =
    A(m; x, u, y) * x .+ B(m; x, u, y) * u .+ Q(m; x, u, y) * e

"State matrix"
A(::Linear; _...) = missing

"Control matrix"
B(::Linear; _...) = 0

Q(::Linear; _...) = 0

"Model size"
xsize(m::Linear) = size(A(m), 2)
esize(m::Linear) = size(Q(m), 2)
usize(m::Linear) = size(B(m), 2)
ysize(m::Linear) = size(A(m) ,1)

cov(m::Model; x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    Q(m; x, u, y) * Q(m; x, u, y)'

"Covariance propagation common formula for linear
and linearized systems"
(m::Model)(P::AbstractMatrix, x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    A(m; x, u, y) * P * A(m; x, u, y)' .+ cov(m; x, u, y) # broadcasting for scalar A * P * A' case

abstract type Nonlinear <: Model end

"Linearized state matrix"
A(m::Nonlinear; x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    ForwardDiff.jacobian(x->m(;x, u, y), x)

"Linearized control matrix"
B(m::Nonlinear; x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    ForwardDiff.jacobian(u->m(;x, u, y), u)

Q(m::Nonlinear; x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    ForwardDiff.jacobian(e->m(;x, u, y, e), zeros(esize(m)))

esize(m::Nonlinear) = missing
