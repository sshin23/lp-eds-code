using JuMP, HiGHS, Random, Plots, LaTeXStrings

default(
    fontfamily="Computer modern",
    label="" # only explicit legend entries
)

function create_ocp_model(
    T, a, b;
    quiet = true, initial = true, terminal = false
    )
    
    # create model
    m = Model(HiGHS.Optimizer)
    set_optimizer_attribute(m, "output_flag", !quiet)

    # variables
    @variable(m, s[1:T])
    @variable(m, u[1:T])

    # auxiliary variables
    @variable(m, sm[1:T] >= 0)
    @variable(m, sp[1:T] >= 0)
    @variable(m, um[1:T] >= 0)
    @variable(m, up[1:T] >= 0)

    @constraint(m, [t=1:T], sm[t] >=-s[t])
    @constraint(m, [t=1:T], sp[t] >= s[t])
    @constraint(m, [t=1:T], um[t] >=-u[t])
    @constraint(m, [t=1:T], up[t] >= u[t])

    initial  && @constraint(m, c1, s[1] == 0)
    @constraint(m, c[t=2:T], s[t] == a*s[t-1] + b*u[t-1])
    terminal && @constraint(m, cT, 0 == a*s[T] + b*u[T])
    
    @objective(m, Min, sum(sm[t] + sp[t] + um[t] + up[t] for t=1:T))
    
    return m
end

function set_ocp_data!(m, g, hs, hu; T = length(g))
    set_normalized_rhs(m[:c1], g[1])
    for t=2:T
        set_normalized_rhs(m[:c][t], g[t])
    end
    for t=1:T
        set_objective_coefficient(m, m[:sm][t], hs[t])
        set_objective_coefficient(m, m[:sp][t], hs[t])
        set_objective_coefficient(m, m[:um][t], hu[t])
        set_objective_coefficient(m, m[:up][t], hu[t])
    end
end

function set_perturbed_data!(v,v_ref,t,sig)
    ptb = sig * randn()
    v .= v_ref
    v[t] += ptb
    return ptb
end

function main(; seed = 0)
    
    Random.seed!(seed)

    # problem parameters
    T = 100 # time horizon
    g_ref = randn(T) # reference data
    hs_ref= ones(T) # reference data
    hu_ref= ones(T) # reference data
    g = zeros(T) # perturbed data
    hs= zeros(T) # perturbed data
    hu= zeros(T) # perturbed data

    c = 2
    b = 2

    # save buffer
    Ns = 10 # numer of samples
    sig = 5 # magnitude of perturbation
    tp  = 50 # perturbed point
    solutions = NamedTuple{(:s, :u), Tuple{Vector{Float64}, Vector{Float64}}}[]

    # create problem
    m = create_ocp_model(T, a, b)
    set_ocp_data!(m, g_ref, hs_ref, hu_ref)
    optimize!(m)

    ref_sol = (s=value.(m[:s]), u=value.(m[:u]))
    diff    = (s=zeros(T), u=zeros(T))

    for i=1:Ns
        ptb = set_perturbed_data!(g , g_ref, tp, sig)
        # ptb2 = set_perturbed_data!(hs, hs_ref, tp, sig)
        # ptb3 = set_perturbed_data!(hu, hu_ref, tp, sig)
        
        set_ocp_data!(m, g, hs, hu)
        optimize!(m)

        sol = (s = value.(m[:s]), u = value.(m[:u]))
        
        diff.s .= max.( diff.s, abs.(ref_sol.s .- sol.s) ./ abs(ptb) )
        diff.u .= max.( diff.u, abs.(ref_sol.u .- sol.u) ./ abs(ptb) )
        
        push!(solutions, sol)
    end

    plts = plot(
        0:T-1, diff.s,
        xlim=(0,T-1),
        framestyle = :box,
        xlabel = L"t",
        ylabel = L"|s_t-s_t^{\prime}| / |\Delta g_{t^*}|",
        label = L"|s_t-s_t^{\prime}|"
    )
    vline!(plts, [tp-1], linestyle=:dash, color=:red, label= L"t^*")

    pltu = plot(
        0:T-1, diff.u,
        xlim=(0,T-1),
        framestyle = :box,
        xlabel = L"t",
        ylabel = L"|u_t-u_t^{\prime}| / |\Delta g_{t^*}|",
        label = L"|u_t-u_t^{\prime}|"
    )
    vline!(pltu, [tp-1], linestyle=:dash, color=:red, label= L"t^*")

    savefig(plts, "fig/ocp-s.pdf")
    savefig(pltu, "fig/ocp-u.pdf")

    return 1
end

main()
