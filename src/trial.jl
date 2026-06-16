#
# Copyright (c) 2026 Julian Trommer, Valentin Höpfner, Andreas Hofmann, Josef Kircher, Tobias Thummerer, and contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
    suggest_int(
        trial::Trial,
        name::String,
        low::T,
        high::T;
        step::T=1,
        log::Bool=false
    ) where {T<:Signed}

Suggest an integer value for the given parameter name within the specified range.
For further information see the [suggest_int](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int) in the Optuna python documentation.


## Arguments
- `trial::Trial`: The trial to suggest the parameter for. (see [Trial](@ref))
- `name::String`: The name of the parameter to suggest.
- `low::T`: The lower bound of the range (inclusive).
- `high::T`: The upper bound of the range (inclusive).

## Keyword Arguments
- `step::T=1`: The step size for the range. The suggested value will be a multiple of `step` away from `low`.
- `log::Bool=false`: If `true`, the range will be sampled on a logarithmic scale. (

## Returns
- `T`: Suggested integer value.
"""
function suggest_int(
    trial::Trial{false}, name::String, low::T, high::T; step::T=1, log::Bool=false
) where {T<:Signed}
    @assert !(step != 1 && log) "The parameters `step` and `log` cannot be used " *
        "at the same time when suggesting an integer."
    return pyconvert(T, trial.trial.suggest_int(name, low, high; step=step, log=log))
end
function suggest_int(
    trial::Trial{true}, name::String, low::T, high::T; step::T=1, log::Bool=false
) where {T<:Signed}
    @assert !(step != 1 && log) "The parameters `step` and `log` cannot be used " *
        "at the same time when suggesting an integer."
    thread_safe() do
        return pyconvert(T, trial.trial.suggest_int(name, low, high; step=step, log=log))
    end
end

"""
    suggest_float(
        trial::Trial,
        name::String,
        low::T,
        high::T;
        step::Union{Nothing,T}=nothing,
        log::Bool=false,
    ) where {T<:AbstractFloat}

Suggest a float value for the given parameter name within the specified range.
For further information see the [suggest_float](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float) in the Optuna python documentation.

## Arguments
- `trial::Trial`: The trial to suggest the parameter for. (see [Trial](@ref))
- `name::String`: The name of the parameter to suggest.
- `low::T`: The lower bound of the range (inclusive).
- `high::T`: The upper bound of the range (inclusive).

## Keyword Arguments
- `step::Union{Nothing,T}=nothing`: A step of discretization.
- `log::Bool=false`: If `true`, the range will be sampled on a logarithmic scale. (

## Returns
- `Float64`: Suggested float value.
"""
function suggest_float(
    trial::Trial,
    name::String,
    low::T,
    high::T;
    step::Union{Nothing,T}=nothing,
    log::Bool=false,
) where {T<:AbstractFloat}
    @warn "Converting to Float64, because that´s what Optuna uses internally. If you need other Float-Types you need to handle conversion after using `suggest_float`." maxlog =
        5
    return suggest_float(
        trial,
        name,
        Float64(low),
        Float64(high);
        step=convert(Union{Nothing,Float64}, step),
        log=log,
    )
end

function suggest_float(
    trial::Trial{false},
    name::String,
    low::Float64,
    high::Float64;
    step::Union{Nothing,Float64}=nothing,
    log::Bool=false,
)
    @assert !(!isnothing(step) && log) "The parameters `step` and `log` cannot be used " *
        "at the same time when suggesting a float."
    return pyconvert(
        Float64, trial.trial.suggest_float(name, low, high; step=step, log=log)
    )
end

function suggest_float(
    trial::Trial{true},
    name::String,
    low::Float64,
    high::Float64;
    step::Union{Nothing,Float64}=nothing,
    log::Bool=false,
)
    @assert !(!isnothing(step) && log) "The parameters `step` and `log` cannot be used " *
        "at the same time when suggesting a float."
    thread_safe() do
        return pyconvert(
            Float64, trial.trial.suggest_float(name, low, high; step=step, log=log)
        )
    end
end

"""
    suggest_categorical(
        trial::Trial,
        name::String,
        choices::Union{Vector{T},Tuple{Vararg{T}}}
    ) where {T<:Union{Bool,Int,AbstractFloat,String}}

Suggest a categorical value for the given parameter name from the specified choices.
For further information see the [suggest_categorical](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical) in the Optuna python documentation.

## Arguments
- `trial::Trial`: The trial to suggest the parameter for. (see [Trial](@ref))
- `name::String`: The name of the parameter to suggest.
- `choices::Union{Vector{T},Tuple{Vararg{T}}}`: The choices to suggest from.

## Returns
- `T`: Suggested categorical value.
"""
function suggest_categorical(
    trial::Trial{false}, name::String, choices::Union{Vector{T},Tuple{Vararg{T}}}
) where {T<:Union{Bool,Int,AbstractFloat,String}}
    return pyconvert(T, trial.trial.suggest_categorical(name, choices))
end
function suggest_categorical(
    trial::Trial{true}, name::String, choices::Union{Vector{T},Tuple{Vararg{T}}}
) where {T<:Union{Bool,Int,AbstractFloat,String}}
    thread_safe() do
        return pyconvert(T, trial.trial.suggest_categorical(name, choices))
    end
end

"""
    suggest_categorical(
        trial::Trial,
        name::String,
        choices::Union{Vector{T},Tuple{Vararg{T}}}
    ) where {T}

Suggest a categorical value for the given parameter name from the specified choices.
For further information see the [suggest_categorical](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical) in the Optuna python documentation.

## Arguments
- `trial::Trial`: The trial to suggest the parameter for. (see [Trial](@ref))
- `name::String`: The name of the parameter to suggest.
- `choices::Union{Vector{T},Tuple{Vararg{T}}}`: The choices to suggest from.

## Returns
- `T`: Suggested categorical value.
"""
function suggest_categorical(
    trial::Trial{false}, name::String, choices::Union{Vector{T},Tuple{Vararg{T}}}
) where {T}
    choices_str = ["$i|$v" for (i, v) in enumerate(choices)]
    choice = pyconvert(String, trial.trial.suggest_categorical(name, choices_str))
    choice_idx = parse(Int, split(choice, '|')[1])
    return choices[choice_idx]
end
function suggest_categorical(
    trial::Trial{true}, name::String, choices::Union{Vector{T},Tuple{Vararg{T}}}
) where {T}
    thread_safe() do
        choices_str = ["$i|$v" for (i, v) in enumerate(choices)]
        choice = pyconvert(String, trial.trial.suggest_categorical(name, choices_str))
        choice_idx = parse(Int, split(choice, '|')[1])
        return choices[choice_idx]
    end
end

"""
    report(
        trial::Trial,
        value::AbstractFloat,
        step::Int
    )

Report an intermediate value for the given trial at a specific step.
For further information see the [report](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report) in the Optuna python documentation.

## Arguments
- `trial::Trial`: The trial to report the value for. (see [Trial](@ref))
- `value::AbstractFloat`: The intermediate value to report.
- `step::Int`: The step at which the value is reported.
"""
function report(trial::Trial{false}, value::AbstractFloat, step::Int)
    return trial.trial.report(value; step=step)
end
function report(trial::Trial{true}, value::AbstractFloat, step::Int)
    thread_safe() do
        return trial.trial.report(value; step=step)
    end
end

"""
    should_prune(
        trial::Trial
    )

Check if the given trial should be pruned based on the pruner's decision.
For further information see the [should_prune](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune) in the Optuna python documentation.

## Arguments
- `trial::Trial`: The trial to check for pruning. (see [Trial](@ref))

## Returns
- `Bool`: `true` if the trial should be pruned, `false` otherwise.
"""
function should_prune(trial::Trial{false})
    return Bool(trial.trial.should_prune())
end
function should_prune(trial::Trial{true})
    thread_safe() do
        return Bool(trial.trial.should_prune())
    end
end

# Trial Attributes
"""
    trial_number(trial::Trial) -> Int

Get the trial's number, which is consecutive and unique within a study.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Int`: The trial number.
"""
function trial_number(trial::Trial{false})
    return pyconvert(Int, trial.trial.number)
end
function trial_number(trial::Trial{true})
    thread_safe() do
        return pyconvert(Int, trial.trial.number)
    end
end

"""
    trial_params(trial::Trial) -> Dict{String,Any}

Get the parameters that were sampled for this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Dict{String,Any}`: Parameter name-value pairs.
"""
function trial_params(trial::Trial{false})
    return pyconvert(Dict{String,Any}, trial.trial.params)
end
function trial_params(trial::Trial{true})
    thread_safe() do
        return pyconvert(Dict{String,Any}, trial.trial.params)
    end
end

"""
    trial_relative_params(trial::Trial) -> Dict{String,Any}

Get the parameters sampled relative to the sampler's search space for this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Dict{String,Any}`: Parameter name-value pairs sampled relative to the search space.
"""
function trial_relative_params(trial::Trial{false})
    return pyconvert(Dict{String,Any}, trial.trial.relative_params)
end
function trial_relative_params(trial::Trial{true})
    thread_safe() do
        return pyconvert(Dict{String,Any}, trial.trial.relative_params)
    end
end

"""
    trial_user_attrs(trial::Trial) -> Dict{String,Any}

Get the user attributes attached to this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Dict{String,Any}`: User attribute name-value pairs.
"""
function trial_user_attrs(trial::Trial{false})
    return pyconvert(Dict{String,Any}, trial.trial.user_attrs)
end
function trial_user_attrs(trial::Trial{true})
    thread_safe() do
        return pyconvert(Dict{String,Any}, trial.trial.user_attrs)
    end
end

"""
    trial_system_attrs(trial::Trial) -> Dict{String,Any}

Get the system attributes attached to this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Dict{String,Any}`: System attribute name-value pairs.
"""
function trial_system_attrs(trial::Trial{false})
    return pyconvert(Dict{String,Any}, trial.trial.system_attrs)
end
function trial_system_attrs(trial::Trial{true})
    thread_safe() do
        return pyconvert(Dict{String,Any}, trial.trial.system_attrs)
    end
end

"""
    trial_distributions(trial::Trial) -> Dict{String,Any}

Get the distributions used to sample each parameter for this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Dict{String,Any}`: Parameter name to distribution mapping. Each value is a Python
  `BaseDistribution` object (e.g. `FloatDistribution`, `IntDistribution`,
  `CategoricalDistribution`), accessible via PythonCall.
"""
function trial_distributions(trial::Trial{false})
    return pyconvert(Dict{String,Any}, trial.trial.distributions)
end
function trial_distributions(trial::Trial{true})
    thread_safe() do
        return pyconvert(Dict{String,Any}, trial.trial.distributions)
    end
end

"""
    trial_values(trial::Trial) -> Vector{Float64}

Get the objective values reported for this trial.

## Arguments
- `trial::Trial`: The trial to query. (see [Trial](@ref))

## Returns
- `Vector{Float64}`: Objective values in the order they were reported. Returns an empty
  vector if the trial has not completed yet.
"""
function trial_values(trial::Trial{false})
    return pyconvert(Vector{Float64}, trial.trial.values)
end
function trial_values(trial::Trial{true})
    thread_safe() do
        return pyconvert(Vector{Float64}, trial.trial.values)
    end
end
