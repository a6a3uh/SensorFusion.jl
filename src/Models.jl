using ForwardDiff

export Model, Linear, Nonlinear, DefaultModel, ZeroModel
export xsize, esize, usize, ysize
export A, B, Q, cov
export model!

abstract type Model{T} end
abstract type Linear{T} <: Model{T} end

function model!(m::Model{T}, z::AbstractVector{T};
                x=zeros(T, xsize(m)),
                u=zeros(T, usize(m)),
                y=zeros(T, ysize(m)),
                e=zeros(T, esize(m))) where T
    
    z .= A(m; x, u, y) * x .+ B(m; x, u, y) * u .+ Q(m; x, u, y) * e
    nothing
end

"Linear system equation"
(m::Linear{T})(;
            x=zeros(T, xsize(m)),
            u=zeros(T, usize(m)),
            y=zeros(T, ysize(m)),
            e=zeros(T, esize(m))) where T =
    A(m; x, u, y) * x .+ B(m; x, u, y) * u .+ Q(m; x, u, y) * e

"State matrix"
A(::Linear; _...) = missing

"Control matrix"
B(m::Linear{T}; _...) where T = T(0)#zeros(ysize(m), usize(m))

"Noise matrix"
Q(m::Linear{T}; _...) where T = T(0)# zeros(ysize(m), esize(m))

"State size"
xsize(m::Linear) = size(A(m), 2)

"Noise size"
esize(m::Linear) = size(Q(m), 2)

"Control size"
usize(m::Linear) = size(B(m), 2)

"Output size"
ysize(m::Linear) = xsize(m)

cov(m::Model; x=zeros(xsize(m)), u=zeros(usize(m)), y=zeros(ysize(m))) =
    Q(m; x, u, y) * Q(m; x, u, y)'

function model!(m::Model{T}, Z::AbstractMatrix,
                P::AbstractMatrix,
                x=zeros(T, xsize(m)),
                u=zeros(T, usize(m)),
                y=zeros(T, ysize(m))) where T
    Z .= A(m; x, u, y) * P * A(m; x, u, y)' .+
        cov(m; x, u, y) # broadcasting for scalar A * P * A' case
    nothing
end

"Covariance propagation common formula for linear
and linearized systems"
function (m::Model{T})(
    P::AbstractMatrix,
    x=zeros(T, xsize(m)),
    u=zeros(T, usize(m)),
    y=zeros(T, ysize(m))) where T
    A(m; x, u, y) * P * A(m; x, u, y)' .+
        cov(m; x, u, y) # broadcasting for scalar A * P * A' case
end

abstract type Nonlinear{T} <: Model{T} end

struct DefaultModel{T} <: Nonlinear{T} end
(::DefaultModel)(;x, _...) = x

struct ZeroModel{T} <: Nonlinear{T} end
(m::ZeroModel{T})(;_...) where T = zeros(T, xsize(m))

"Linearized state matrix"
A(m::Nonlinear{T};
  x=zeros(T, xsize(m)),
  u=zeros(T, usize(m)),
  y=zeros(T, ysize(m))) where T=
    ForwardDiff.jacobian(x->m(;x, u, y), x)

"Linearized control matrix"
B(m::Nonlinear{T};
  x=zeros(T, xsize(m)),
  u=zeros(T, usize(m)),
  y=zeros(T, ysize(m))) where T =
    ForwardDiff.jacobian(u->m(;x, u, y), u)

Q(m::Nonlinear{T};
  x=zeros(T, xsize(m)),
  u=zeros(T, usize(m)),
  y=zeros(T, ysize(m))) where T =
    ForwardDiff.jacobian(e->m(;x, u, y, e), zeros(esize(m)))

esize(m::Nonlinear) = missing
