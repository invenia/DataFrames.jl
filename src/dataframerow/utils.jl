# Rows grouping.
# Maps row contents to the indices of all the equal rows.
# Used by groupby(), join(), nonunique()
struct RowGroupDict{T<:AbstractDataFrame}
    "source data table"
    df::T
    "number of groups"
    ngroups::Int
    "indices of group-representative rows"
    gslots::Vector{Int}
    "group index for each row"
    groups::Vector{Int}
    "permutation of row indices that sorts them by groups"
    rperm::Vector{Int}
    "starts of ranges in rperm for each group"
    starts::Vector{Int}
    "stops of ranges in rperm for each group"
    stops::Vector{Int}
end

"""
    row_group_slot

Helper function for RowGroupDict.

Returns a tuple:
1) the number of row groups in a data table
2) indices of group-representative rows, non-zero values are
    the indices of the first row in a group

Optional group vector is set to the group indices of each row
"""
function row_group_slots(df::AbstractDataFrame,
                         groups::Union{Vector{Int}, Nothing} = nothing,
                         skipmissing::Bool = false)

    missings = fill(false, skipmissing ? nrow(df) : 0)
    if skipmissing
        for col in columns(df)
            @inbounds for i in eachindex(col)
                el = col[i]
                # el isa Missing should be redundant
                # but it gives much more efficient code on Julia 0.6
                missings[i] |= (el isa Missing || ismissing(el))
            end
        end
    end

    row_group_slots(ntuple(i -> df[i], ncol(df)), missings, groups, skipmissing)
end

function row_group_slots(cols::Tuple{Vararg{AbstractVector}},
                         missings::AbstractVector{Bool},
                         groups::Union{Vector{Int}, Nothing} = nothing,
                         skipmissing::Bool = false)
    @assert groups === nothing || length(groups) == length(cols[1])

    nrows = length(cols[1])
    # Contains the indices of the starting row for each group
    gslots = zeros(Int, nrows)
    # If missings are to be skipped, they will all go to group 1
    ngroups = skipmissing ? 1 : 0

    @inbounds for i in 1:nrows
        # Use 0 for non-missing values to catch bugs if group is not found
        gix = skipmissing && missings[i] ? 1 : 0
        # Skip rows contaning at least one missing (assigning them to group 0)
        if !skipmissing || !missings[i]

            @inbounds for g_row in gslots # only need to compare againt indices that start groups
                if g_row == 0 # unoccupied slot, current row starts a new group
                    gix = ngroups += 1
                    gslots[skipmissing ? ngroups - 1 : ngroups] = i
                    break
                elseif isequivalent_row(cols, i, g_row) # hit, matches a previous group
                    gix = groups !== nothing ? groups[g_row] : 0
                    break
                end
            end
        end
        if groups !== nothing
            groups[i] = gix
        end
    end
    return ngroups, gslots
end

# Builds RowGroupDict for a given DataFrame.
# Partly uses the code of Wes McKinney's groupsort_indexer in pandas (file: src/groupby.pyx).
function group_rows(df::AbstractDataFrame, skipmissing::Bool = false)
    groups = Vector{Int}(undef, nrow(df))
    ngroups, gslots = row_group_slots(df, groups, skipmissing)

    # count elements in each group
    stops = zeros(Int, ngroups)
    @inbounds for g_ix in groups
        stops[g_ix] += 1
    end

    # group start positions in a sorted table
    starts = Vector{Int}(undef, ngroups)
    if !isempty(starts)
        starts[1] = 1
        @inbounds for i in 1:(ngroups-1)
            starts[i+1] = starts[i] + stops[i]
        end
    end

    # define row permutation that sorts them into groups
    rperm = Vector{Int}(undef, length(groups))
    copyto!(stops, starts)
    @inbounds for (i, gix) in enumerate(groups)
        rperm[stops[gix]] = i
        stops[gix] += 1
    end
    stops .-= 1

    # drop group 1 which contains rows with missings in grouping columns
    if skipmissing
        splice!(starts, 1)
        splice!(stops, 1)
        ngroups -= 1
    end

    return RowGroupDict(df, ngroups, gslots, groups, rperm, starts, stops)
end

# Find index of a row in gd that matches given row by content, 0 if not found
function findrow(gd::RowGroupDict,
                 df::DataFrame,
                 gd_cols::Tuple{Vararg{AbstractVector}},
                 df_cols::Tuple{Vararg{AbstractVector}},
                 row::Int)
    (gd.df === df) && return row # same table, return itself
    # different tables, content matching required
    @inbounds for g_row in 1:length(gd_cols[1])
        isequivalent_row(gd_cols, g_row, df_cols, row) && return g_row  # hit
        # miss, try the next slot
    end
    return 0 # not found
end

# Find indices of rows in 'gd' that match given row by content.
# return empty set if no row matches
function findrows(gd::RowGroupDict,
                  df::DataFrame,
                  gd_cols::Tuple{Vararg{AbstractVector}},
                  df_cols::Tuple{Vararg{AbstractVector}},
                  row::Int)
    g_row = findrow(gd, df, gd_cols, df_cols, row)
    (g_row == 0) && return view(gd.rperm, 0:-1)
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end

function Base.getindex(gd::RowGroupDict, dfr::DataFrameRow)
    g_row = findrow(gd, dfr.df, ntuple(i -> gd.df[i], ncol(gd.df)),
                    ntuple(i -> dfr.df[i], ncol(dfr.df)), dfr.row)
    (g_row == 0) && throw(KeyError(dfr))
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end
