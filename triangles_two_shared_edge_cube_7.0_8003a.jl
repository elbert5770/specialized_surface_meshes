using LinearAlgebra
using GLMakie
using Statistics
using PlyIO
using ColorSchemes
import ColorSchemes.rainbow
using PrettyTables

# Define a structure to hold mesh data
struct MeshDatatemp
    faces::Matrix{Int64}
    vertices::Matrix{Float64}
    normals::Matrix{Float64}
    centroids::Matrix{Float64}
    triangles::Array{Float64, 3}
end
struct MeshData
    faces::Matrix{Int64}
    vertices::Matrix{Float64}
    normals::Matrix{Float64}
    centroids::Matrix{Float64}
    triangles::Array{Float64, 3}
    neighbors::Matrix{Int64}  # New field: 4 x num_faces array of nearest neighbors
    neighbor_count::Vector{Int64}
    edges::Matrix{Int64}
end

# Define MeshGroup as a vector of MeshData
const MeshGroup = Vector{MeshDatatemp}

# Define a structure to store tetrahedrons
struct Tetrahedron
    vertices::Array{Float64, 2}  # 3x4 array, each column is a vertex
    adjacent::Vector{Int64}        # Vector to store adjacent tetrahedron numbers
end

# Define a structure to hold a collection of 17 Tetrahedron structs
struct TetrahedronGroup
    tetrahedra::NTuple{23, Tetrahedron}
    base_triangle::Array{Float64,2} 
end

# Constructor for TetrahedronGroup
function TetrahedronGroup(tetrahedra::Vector{Tetrahedron},base_triangle::Array{Float64,2})
    if length(tetrahedra) != 23
        throw(ArgumentError("TetrahedronGroup must contain exactly 23 Tetrahedron structs"))
    end
    return TetrahedronGroup(Tuple(tetrahedra),base_triangle)
end

# Helper function to access individual tetrahedra in the group
function get_tetrahedron(group::TetrahedronGroup, index::Int)
    if 1 <= index <= 23
        return group.tetrahedra[index]
    else
        throw(BoundsError(group, index))
    end
end


# Function to calculate the centroid of a tetrahedron
function calculate_centroid(tetra::Tetrahedron)
    vertices = get_vertices(tetra)
    return sum(vertices) / 4
end

# Constructor for Tetrahedron
function Tetrahedron(v1::Vector{Float64}, v2::Vector{Float64}, v3::Vector{Float64}, v4::Vector{Float64})
    return Tetrahedron(hcat(v1, v2, v3, v4), Int[])
end

# Helper function to get vertices of a tetrahedron
function get_vertices(tetra::Tetrahedron)
    return [tetra.vertices[:, i] for i in 1:4]
end

# Helper function to calculate the volume of a tetrahedron
function calculate_volume(tetra::Tetrahedron)
    v1, v2, v3, v4 = get_vertices(tetra)
    return abs(dot(v1 - v4, cross(v2 - v4, v3 - v4))) / 6
end

# Function to add an adjacent tetrahedron
function add_adjacent!(tetra::Tetrahedron, adjacent_num::Int)
    push!(tetra.adjacent, adjacent_num)
end

# Function to get adjacent tetrahedrons
function get_adjacent(tetra::Tetrahedron)
    return tetra.adjacent
end


# Function to read PLY file using PlyIO
function read_ply(filename)
    ply = load_ply(filename)
    
    # Extract vertex data
    vertex_x = ply["vertex"]["x"]
    vertex_y = ply["vertex"]["y"]
    vertex_z = ply["vertex"]["z"]
    vertices = [[x, y, z] for (x, y, z) in zip(vertex_x, vertex_y, vertex_z)]
    
    # Extract face data
    faces = [f.+1 for f in ply["face"]["vertex_indices"]]  # Add 1 to convert to 1-based indexing
    
    return vertices, faces
end



# Function to calculate normal for a triangle
function calculate_normal(v1, v2, v3)
    edge1 = v1 - v2
    edge2 = v3 - v1
    return normalize(cross(edge1, edge2))
end

# Function to calculate centroid of a triangle
function calculate_centroid(v1, v2, v3)
    return (v1 + v2 + v3) / 3
end

# Filter function based on the criteria
function meets_criteria(v)
    return 2504.0 <= v[1] <= 2505.0 &&
           296.0 <= v[2] <= 299.0 &&
           297.0 <= v[3] <= 300.0
end
# function meets_criteria(v)
#     return 7199.0 <= v[1] <= 7212.0 &&
#            248.0 <= v[2] <= 250.0 &&
#            313.0 <= v[3] <= 314.0
# end
# function meets_criteria(v)
#     return 7199.0 <= v[1] <= 7202.0 &&
#            248.0 <= v[2] <= 251.0 &&
#            310.0 <= v[3] <= 311.0
# end


function calculate_normal(v1, v2, v3)
    edge1 = v2 - v1
    edge2 = v3 - v1
    return normalize(cross(edge1, edge2))
end

function shared_edge_vertices(faces::Matrix{Int64})
    num_faces = size(faces, 2)
    edges_with_faces = zeros(Int64,3,num_faces*3)
    counter = 1
    for (i,face) in enumerate(eachcol(faces))
        # @show face
        new_edges = [
            (min(face[1], face[2]), max(face[1], face[2]), i),
            (min(face[2], face[3]), max(face[2], face[3]), i),
            (min(face[3], face[1]), max(face[3], face[1]), i)
        ]
        # @show new_edges
        for j in 1:3
            edges_with_faces[j,counter] = new_edges[1][j]
            edges_with_faces[j,counter+1] = new_edges[2][j]
            edges_with_faces[j,counter+2] = new_edges[3][j]
        end
        # @show edges_with_faces[:,counter:counter+2]
        counter += 3
    end

    # Sort and remove duplicates, keeping only the first occurrence (which includes the face number)
    p1 = sortperm(eachslice(edges_with_faces; dims=2), by=x->(x[1], x[2]))
    # sorted_unique_edges = sort(unique(x -> (x[1], x[2]), edges_with_faces))
    sorted_edges_with_faces = edges_with_faces[:, p1]
# @show sorted_edges_with_faces'
    edges = zeros(Int64, 5, num_faces*3)
    # unshared_edges = zeros(Int64, 3, num_faces*3)
    shared_edge_counter = 0
    unshared_edge_counter = 0
    # Iterate through the sorted edges to find shared edges and unshared faces
    i = 1
    maxsize = size(sorted_edges_with_faces, 2)
    while i <= maxsize
        if i < maxsize && sorted_edges_with_faces[1:2, i] == sorted_edges_with_faces[1:2, i+1]
            shared_edge_counter += 1

            if i < maxsize-1 && sorted_edges_with_faces[1:2, i+2] == sorted_edges_with_faces[1:2, i+1]
                
                edges[:, shared_edge_counter] = [
                    sorted_edges_with_faces[1, i],
                    sorted_edges_with_faces[2, i],
                    sorted_edges_with_faces[3, i],
                    sorted_edges_with_faces[3, i+1],
                    sorted_edges_with_faces[3, i+2]
                ]
                i += 3
            else
                edges[:, shared_edge_counter] = [
                    sorted_edges_with_faces[1, i],
                    sorted_edges_with_faces[2, i],
                    sorted_edges_with_faces[3, i],
                    sorted_edges_with_faces[3, i+1],
                    0
                ]
                i += 2
            end
              # Skip the next edge as it's part of the shared pair
        else
            shared_edge_counter += 1
            edges[:,shared_edge_counter] = [
                sorted_edges_with_faces[1, i],
                sorted_edges_with_faces[2, i],
                sorted_edges_with_faces[3, i],
                0,
                0
            ]   
            i += 1
        end
    end
    
    # Resize the edges array to remove unused columns
    edges = edges[:, 1:shared_edge_counter]
    
    # Convert unshared_faces set to an array
    # unshared_edges = unshared_edges[:, 1:unshared_edge_counter]
    
    # @show size(edges)
    # @show edges'
    # @show size(unshared_edges)
    # @show unshared_edges'
    return edges
end

function merge_mesh_data(mesh_group::MeshGroup)
    all_faces = Vector{Vector{Int64}}()
    all_vertices = Vector{Vector{Float64}}()
    all_normals = Vector{Vector{Float64}}()
    all_centroids = Vector{Vector{Float64}}()
    all_triangles = Vector{Matrix{Float64}}()

    vertex_map = Dict{Vector{Float64}, Int64}()
    centroid_map = Dict{Vector{Float64}, Int64}()

    for mesh_data in mesh_group
        for i in 1:size(mesh_data.centroids, 2)
            centroid = mesh_data.centroids[:, i]
            centroid_key = round.(centroid, digits=10)  # Round to machine precision

            if !haskey(centroid_map, centroid_key)
                centroid_map[centroid_key] = length(all_centroids) + 1
                push!(all_centroids, centroid)
                push!(all_normals, mesh_data.normals[:, i])
                push!(all_triangles, mesh_data.triangles[:, :, i])

                new_face = Int64[]
                for j in 1:3
                    vertex = mesh_data.vertices[:, mesh_data.faces[j, i]]
                    vertex_key = round.(vertex, digits=10)
                    if !haskey(vertex_map, vertex_key)
                        vertex_map[vertex_key] = length(all_vertices) + 1
                        push!(all_vertices, vertex)
                    end
                    push!(new_face, vertex_map[vertex_key])
                end
                push!(all_faces, new_face)
            end
        end
    end

    # Convert to appropriate array types
    faces = hcat(all_faces...)
    vertices = hcat(all_vertices...)
    normals = hcat(all_normals...)
    centroids = hcat(all_centroids...)
    triangles = cat(all_triangles..., dims=3)
    edges = shared_edge_vertices(faces::Matrix{Int64})
    # Calculate neighbors
    num_faces = size(faces, 2)
    neighbors = zeros(Int64, 5, num_faces)
    neighbor_count = zeros(Int64, num_faces)
    for (i,edge) in enumerate(eachcol(edges))
        @show edge
        face1 = edge[3]
        face2 = edge[4]
        face3 = edge[5]
        if face1 != 0 && face2 != 0
            neighbor_count[face1] += 1
            neighbor_count[face2] += 1
            neighbors[neighbor_count[face1],face1] = face2
            neighbors[neighbor_count[face2],face2] = face1
            if face3 != 0
                neighbor_count[face1] += 1
                neighbor_count[face2] += 1
                neighbor_count[face3] += 1
                neighbors[neighbor_count[face1],face1] = face3
                neighbors[neighbor_count[face2],face2] = face3
                neighbors[neighbor_count[face3],face3] = face1
                neighbor_count[face3] += 1
                neighbors[neighbor_count[face3],face3] = face2
            end
        end
    
        
    end
    # @show neighbors'
    # @show neighbor_count
    for (face_idx, neighbor_count) in enumerate(neighbor_count)
        # Debug output
        if neighbor_count == 0
            println("Face $face_idx has no neighbors")
        else
            println("Face $face_idx has $neighbor_count neighbors")

        end
    end

    # Debug output
    println("Total faces: ", num_faces)
    println("Faces with 3 neighbors: ", count(i -> count(!iszero, neighbors[:, i]) == 3, 1:num_faces))
    println("Faces with 4 neighbors: ", count(i -> count(!iszero, neighbors[:, i]) == 4, 1:num_faces))
    println("Faces with 5 neighbors: ", count(i -> count(!iszero, neighbors[:, i]) == 5, 1:num_faces))
    return MeshData(faces, vertices, normals, centroids, triangles, neighbors,neighbor_count,edges)
end


function calculate_vertex_normals(mesh::MeshData)
    vertex_normals = Dict{Int64, Vector{Vector{Float64}}}()
    vertex_faces = Dict{Int64, Set{Int64}}()
    vertex_edges = Dict{Int64, Set{Int64}}()
    for (face_idx, face) in enumerate(eachcol(mesh.faces))
        for vertex_idx in face
            if !haskey(vertex_faces, vertex_idx)
                vertex_faces[vertex_idx] = Set{Int64}()
            end
            push!(vertex_faces[vertex_idx], face_idx)
        end
    end
    @show vertex_faces
    for (edge_idx, edge) in enumerate(eachcol(mesh.edges))
        for vertex_idx in edge[1:2]
            if !haskey(vertex_edges, vertex_idx)
                vertex_edges[vertex_idx] = Set{Int64}()
            end
            push!(vertex_edges[vertex_idx], edge_idx)
        end
    end
    @show vertex_edges
    for (vertex_idx, edge_set) in vertex_edges
        normals = [mesh.normals[:, face_idx] for face_idx in vertex_faces[vertex_idx]]
        state = false
        split_edge = 0
        edge_midpoint = Vector{Float64}(undef,3)
        for edge in edge_set
            if mesh.edges[5,edge] > 0
                state = true
                split_edge = edge
                edge_midpoint = (mesh.vertices[:,mesh.edges[1,edge]] + mesh.vertices[:,mesh.edges[2,edge]]) / 2
            end
        end
        # for edge in edge_set
            if state == true
                vertex_normals[vertex_idx] = [normalize(mesh.normals[:,mesh.edges[3,split_edge]]+mesh.normals[:,mesh.edges[4,split_edge]]),normalize(-mesh.normals[:,mesh.edges[4,split_edge]]+mesh.normals[:,mesh.edges[5,split_edge]]),normalize(-mesh.normals[:,mesh.edges[3,split_edge]]-mesh.normals[:,mesh.edges[5,split_edge]]),edge_midpoint]
            else
                vertex_normals[vertex_idx] = [normalize(sum(normals))]
            end
            
        # end
        
    end

    return vertex_normals
end

function select_normal(vertex_normals, vertex_idx, centroid, vertex,face_normal,reverse,face,face_idx)
    @show face,face_idx
    normals = vertex_normals[vertex_idx]
    max_idx = 0
    min_value = -Inf
    max_value = -Inf
    edge_midpoint = Vector{Float64}(undef,3)
    face_normal = face_normal.*0.25
    if length(normals) == 1
        return normals[1]
    else
        # Select the normal that's closest to pointing towards the centroid
        # @show vertex_idx,normals,face_normal,dot.(normals, Ref(face_normal).*reverse)
        edge_midpoint = normals[4]
        for i in 1:3
            normal = normals[i]
            # @show i,norm((i+vertex[:,vertex_idx]) - (Ref(face_normal).*reverse +centroid))
            @show vertex_idx,vertex,normals,face_normal,reverse,centroid,face_normal.*reverse.+centroid,edge_midpoint
            @show (centroid+face_normal*reverse)-edge_midpoint
            @show centroid-edge_midpoint
            @show dot(face_normal,normal)
            # theta = acos(dot(edge_midpoint-vertex,centroid-vertex)/norm(edge_midpoint-vertex)/norm(centroid-vertex))
            # @show theta
            # direction = sin(theta)*(centroid-vertex)
            # @show direction
            projection = dot(centroid+face_normal*reverse-vertex,edge_midpoint-vertex)
            new_mid = normalize(edge_midpoint-vertex).*projection
            # min_value = norm((normal .+vertex) .- (face_normal .*reverse .+centroid))
            # max_value = dot(normal, (centroid+face_normal*reverse)-edge_midpoint)#/norm(centroid+face_normal*reverse-edge_midpoint)/norm(normal)
            # max_value += dot(normal, centroid-edge_midpoint)
            # max_value = dot(normal, edge_midpoint.-centroid+face_normal*reverse)
            # max_value += dot(normal, (centroid+face_normal*reverse)-vertex)
            # max_value = 0#dot(normal,centroid+face_normal*reverse-new_mid)#/norm(centroid+face_normal*reverse-new_mid)/norm(normal)
             
            max_value_temp = dot(normal, (centroid+face_normal*reverse)-edge_midpoint)/norm(centroid+face_normal*reverse-edge_midpoint)/norm(normal)
            if isnan(max_value_temp) == false
                max_value = max_value_temp
            end
            
            if max_value > min_value && dot(face_normal*reverse,normal) > 0 
                min_value = max_value
                max_idx = i
            end
            @show max_value,i,vertex_idx,max_idx
        end
        # end
        return normals[max_idx].*reverse
    end
end

# Function to calculate the plane equation (ax + by + cz + d = 0) given three points
function get_plane_equation(p1, p2, p3)
    v1 = p2 - p1
    v2 = p3 - p1
    n = cross(v1, v2)
    a, b, c = n
    d = -dot(n, p1)
    return a, b, c, d
end

# Function to check if a point is on a plane (within a tolerance)
function is_on_plane(point, plane, tol=1e-3)
    a, b, c, d = plane
    return isapprox(a*point[1] + b*point[2] + c*point[3] + d, 0, atol=tol)
end

function calculate_prism_vertices(tetrahedron_group)
    all_vertices = vcat([get_vertices(tetra) for tetra in tetrahedron_group.tetrahedra]...)
    unique_vertices = unique(all_vertices)
    
    # Sort vertices to ensure consistent ordering
    sorted_vertices = sort(unique_vertices, by=v->(v[1], v[2], v[3]))
    
    # The first three vertices form one triangular face
    # The last three vertices form the other triangular face
    return sorted_vertices[1:3], sorted_vertices[end-2:end]
end

function get_exterior_surface(face_centroid, prism_vertices)
    base1, base2 = prism_vertices
    v1, v2, v3 = base1
    v4, v5, v6 = base2

    # Calculate plane equations for each face
    base1_plane = get_plane_equation(v1, v2, v3)
    base2_plane = get_plane_equation(v4, v5, v6)

    # Check if the centroid is on one of the base planes
    if is_on_plane(face_centroid, base1_plane)
        return "Base1"
    elseif is_on_plane(face_centroid, base2_plane)
        return "Base2"
    end

    # If not on a base, check which side plane it's on
    edges = [(v1, v4), (v2, v5), (v3, v6)]
    for (i, (e1, e2)) in enumerate(edges)
        edge_vector = e2 - e1
        edge_midpoint = (e1 + e2) / 2
        to_centroid = face_centroid - edge_midpoint
        
        # Check if the centroid is close to the plane defined by this edge
        if abs(dot(normalize(cross(edge_vector, to_centroid)), normalize(edge_vector))) < 1e-3
            return "Side$i"
        end
    end

    return "Interior"
end

# Helper function to get faces of a tetrahedron
function get_faces(tetra::Tetrahedron)
    vertices = get_vertices(tetra)
    return [
        hcat(vertices[1], vertices[2], vertices[3]),
        hcat(vertices[1], vertices[2], vertices[4]),
        hcat(vertices[1], vertices[3], vertices[4]),
        hcat(vertices[2], vertices[3], vertices[4])
    ]
end

# Helper function to check if two faces are approximately equal
function are_faces_equal(face1, face2)
    return all(j -> any(k -> all(isapprox.(face1[:, j], face2[:, k], atol=1e-6)), 1:3), 1:3)
end

function calculate_normal(face)
    v1 = face[:, 1]
    v2 = face[:, 2]
    v3 = face[:, 3]
    @show v1,v2,v3
    edge1 = v2 .- v1
    edge2 = v3 .- v1
    return normalize(cross(edge1, edge2))
end

function get_face_vertices(tetra, face_indices)
    vertices = get_vertices(tetra)
    return [vertices[face_indices[1]], vertices[face_indices[2]], vertices[face_indices[3]]]
end

# Modify the reconstruct_prism function:
function reconstruct_prism(unmatched_faces)
    prism_faces = []
    
    # Group faces by their normal vectors (or their negatives)
    grouped_faces = Dict()
    for face_info in unmatched_faces
        tetra_index, face_index, face = face_info
        normal = calculate_normal(face)
        centroid = calculate_centroid(face)
        key = (round.(abs.(normal), digits=6)...,)  # Use absolute values for grouping
        if haskey(grouped_faces, key)
            push!(grouped_faces[key], (face_info, normal, centroid))
        else
            grouped_faces[key] = [(face_info, normal, centroid)]
        end
    end
    @show grouped_faces
    # Identify matching faces
    for (key, faces) in grouped_faces
        if length(faces) == 2
            (face1, normal1, centroid1), (face2, normal2, centroid2) = faces
            centroid_vector = centroid2 - centroid1
            if isapprox(abs(dot(normalize(centroid_vector), normal1)), 1, atol=1e-6)
                push!(prism_faces, face1[1:2])
                push!(prism_faces, face2[1:2])
            end
        elseif length(faces) == 3
            # This is likely a triangular base
            push!(prism_faces, faces[1][1][1:2])
        end
    end
    
    return prism_faces
end

function calculate_centroid(face)
    return sum(face, dims=2) ./ 3
end

function main()
    f = Figure(backgroundcolor = :black)
    # ax4 = [GLMakie.LScene(f[i, j]) for i in 1:2 for j in 1:2]
    ax = GLMakie.LScene(f[1, 1])
    meshcolor = [:white,:red,:green]
    plycolor = [:yellow,:pink,:lime]
    linestyle = [:dashdot,:dot,:dash,:solid]
    tet_color = [:lime,:yellow,:red,:orange,:white,:magenta]
    linewidth = [1,2,1,2,3,1,2,3,1]
    arrow_scale = 0.25
    normal_color = :green
    normal_linewidth = 0.02
    mesh_names = ["8003_1_2250_250_4.ply","8003_1_2750_250_4.ply"]
    num_meshes = length(mesh_names)
    
    tetrahedrons_cells = Vector{Vector{TetrahedronGroup}}(undef,num_meshes)
    mesh_group = MeshGroup(undef,num_meshes)
    for (meshindex,mesh) in enumerate(mesh_names)
        # Read PLY file
        vertices, faces = read_ply(mesh)
        # Filter vertices
        filtered_vertex_indices = findall(meets_criteria, vertices)
        filtered_vertices = vertices[filtered_vertex_indices]

        # Filter faces and create a mapping for new vertex indices
        old_to_new_index = Dict(old => new for (new, old) in enumerate(filtered_vertex_indices))
        filtered_faces = []
        for face in faces
            if all(v -> v in filtered_vertex_indices, face)
                new_face = [old_to_new_index[v] for v in face]
                push!(filtered_faces, new_face)
            end
        end

        # Process triangles



        # @show size(filtered_faces)
        faces = zeros(Int64, 3, length(filtered_faces))
        vertices = zeros(Float64, 3, length(filtered_vertices))
        normals = zeros(Float64, 3, length(filtered_faces))
        centroids = zeros(Float64, 3, length(filtered_faces))
        triangles = zeros(Float64, 3, 3, length(filtered_faces))
        for (i, face) in enumerate(filtered_faces)
            v1, v2, v3 = filtered_vertices[face]
            
            
            # @show v1,v2,v3,face
            for j in 1:3
                triangles[j, :, i] .= [v1[j], v2[j], v3[j]]
            end
            normals[:,i] .= calculate_normal(v1, v2, v3)
            centroids[:,i] .= calculate_centroid(v1, v2, v3)
            faces[:, i] .= face
        end
        for (i, vertex) in enumerate(filtered_vertices)
            vertices[:, i] .= vertex
        end
        mesh_data = MeshDatatemp(faces, vertices, normals, centroids, triangles)
        mesh_group[meshindex] = mesh_data
    end
    unified_mesh = merge_mesh_data(mesh_group)
    # @show unified_mesh.faces,unified_mesh.neighbors
    # Debug output
    println("Number of faces: ", size(unified_mesh.faces, 2))
    println("Number of vertices: ", size(unified_mesh.vertices, 2))
    println("Shape of neighbors array: ", size(unified_mesh.neighbors))


    # Now use unified_mesh for further processing or visualization
    @show size(unified_mesh.faces)
    @show size(unified_mesh.vertices)
    @show size(unified_mesh.normals)
    @show size(unified_mesh.centroids)
    @show size(unified_mesh.triangles)
    # Plot triangles from unified_mesh
    # for i in 1:size(unified_mesh.triangles, 3)
    #     triangle = unified_mesh.triangles[:, :, i]
    #     x = [triangle[1, j] for j in 1:3]
    #     y = [triangle[2, j] for j in 1:3]
    #     z = [triangle[3, j] for j in 1:3]
        
    #     # Draw the triangle
    #     GLMakie.lines!(ax, [x; x[1]], [y; y[1]], [z; z[1]], 
    #                    color = rainbow[mod(i, 17)+1], linewidth = linewidth[1])
                                   
    #         centroid_x = sum(x) / 3
    #         centroid_y = sum(y) / 3
    #         centroid_z = sum(z) / 3

    #         # Add text label for the face number
    #         GLMakie.text!(ax, "$i",
    #             position = (centroid_x, centroid_y, centroid_z),
    #             color = :white,
    #             align = (:center, :center),
    #             fontsize = 10)  # Adjust textsize as needed
    # # Label vertices
    # # for (vertex_idx, vertex) in enumerate(eachcol(unified_mesh.vertices))
    # #     GLMakie.text!(ax, "v$vertex_idx",
    # #         position = (vertex[1], vertex[2], vertex[3]),
    # #         color = :red,
    # #         align = (:center, :center),
    # #         fontsize = 24)  # Adjust fontsize as needed
    # # end
    # # Plot edge labels
    # # for (edge_idx, edge) in enumerate(eachcol(unified_mesh.edges))
    # #     v1, v2 = edge[1:2]
    # #     if v1 != 0 && v2 != 0
    # #         # Calculate midpoint of the edge
    # #         midpoint = (unified_mesh.vertices[:, v1] + unified_mesh.vertices[:, v2]) / 2
            
    # #         # Add text label for the edge number
    # #         GLMakie.text!(ax, "e$edge_idx",
    # #             position = (midpoint[1], midpoint[2], midpoint[3]),
    # #             color = :green,
    # #             align = (:center, :center),
    # #             fontsize = 18)  # Adjust fontsize as needed
    # #     end
    # # end
            
    # end
    # GLMakie.cam3d!(ax)
    # display(f)
    # for (meshindex,mesh_data) in enumerate(unified_mesh)
    meshindex = 1

    # Calculate vertex normals
    vertex_normals = calculate_vertex_normals(unified_mesh)

    # Debug output
    multiple_normal_vertices = [v for (v, normals) in vertex_normals if length(normals) > 1]
    println("Number of vertices with multiple normals: ", length(multiple_normal_vertices))
    for v in multiple_normal_vertices
        println("Vertex $v has $(length(vertex_normals[v])) normals")
    end

    # Create new triangles from vertex normals and store selected normals
    front_triangles = zeros(Float64, 3, 3, size(unified_mesh.faces, 2))
    back_triangles = zeros(Float64, 3, 3, size(unified_mesh.faces, 2))
    selected_normals = zeros(Float64, 3, 3, size(unified_mesh.faces, 2))
    
    for (i, face) in enumerate(eachcol(unified_mesh.faces))
        centroid = unified_mesh.centroids[:, i]
        # edge_midpoint = (unified_mesh.vertices[:, unified_mesh.edges[1,unified_mesh.edges[5,i]]] + unified_mesh.vertices[:, unified_mesh.edges[2,unified_mesh.edges[5,i]]) / 2
        for j in 1:3
            vertex_idx = face[j]
            vertex = unified_mesh.vertices[:, vertex_idx]
            reverse = 1
                        
            normal = select_normal(vertex_normals, vertex_idx, centroid, vertex,unified_mesh.normals[:,i],reverse,face,i)
            front_triangles[:, j, i] = vertex + normal * arrow_scale
            reverse = -1
            normal = select_normal(vertex_normals, vertex_idx, centroid, vertex,unified_mesh.normals[:,i],reverse,face,i)
            back_triangles[:, j, i] = vertex - normal * arrow_scale
            selected_normals[:, j, i] = normal
        end
    end
    num_original_triangles = size(unified_mesh.triangles, 3)
    midpoint_triangles = zeros(Float64, 3, 3, num_original_triangles)
    front_midpoint_triangles = zeros(Float64, 3, 3, num_original_triangles)
    back_midpoint_triangles = zeros(Float64, 3, 3, num_original_triangles)

    for i in 1:num_original_triangles
        v1, v2, v3 = eachcol(view(unified_mesh.triangles, :, :, i))
        
        # Calculate midpoints
        m1 = (v1 + v2) / 2
        m2 = (v2 + v3) / 2
        m3 = (v3 + v1) / 2
        
        # Create new triangle from midpoints
        midpoint_triangles[:, :, i] = [m1 m2 m3]

        v1, v2, v3 = eachcol(view(front_triangles, :, :, i))
        
        # Calculate midpoints
        m1 = (v1 + v2) / 2
        m2 = (v2 + v3) / 2
        m3 = (v3 + v1) / 2
        front_midpoint_triangles[:, :, i] = [m1 m2 m3]

        v1, v2, v3 = eachcol(view(back_triangles, :, :, i))
        
        # Calculate midpoints
        m1 = (v1 + v2) / 2
        m2 = (v2 + v3) / 2
        m3 = (v3 + v1) / 2
        back_midpoint_triangles[:, :, i] = [m1 m2 m3]
    end

    @show size(midpoint_triangles)
    # # Plot front and back triangles
    # for i in 1:size(front_triangles, 3)
    #     for (triangles, color) in [(front_triangles, :red) (back_triangles, :blue)]
    #         triangle = triangles[:, :, i]
    #         x = [triangle[1, j] for j in 1:3]
    #         y = [triangle[2, j] for j in 1:3]
    #         z = [triangle[3, j] for j in 1:3]
            
    #         GLMakie.lines!(ax, [x; x[1]], [y; y[1]], [z; z[1]], 
    #                         color = rainbow[mod(i, 17)+1], linewidth = linewidth[1])
    #     end
    # end
    # @show vertex_normals
    # Plot all normals as arrows
    # for (vertex_idx, normals) in vertex_normals
    #     vertex = unified_mesh.vertices[:, vertex_idx]
    #     for (i,normal) in enumerate(normals)
    #         if i < 4
    #             end_point = vertex + normal * arrow_scale

    #         GLMakie.arrows!(ax, 
    #             [vertex[1]], [vertex[2]], [vertex[3]],
    #             [normal[1]], [normal[2]], [normal[3]],
    #             color = normal_color, linewidth = normal_linewidth,
    #             arrowsize = 0.1)
    #         else
    #             GLMakie.text!(ax, "e$i",
    #             position = (vertex[1], vertex[2], vertex[3]),
    #             color = :green,
    #             align = (:center, :center),
    #             fontsize = 18)  # Adjust fontsize as needed
    #         end
    #     end
    # end
    # # Display face normals
    # for i in 1:size(unified_mesh.faces, 2)
    #     face_center = unified_mesh.centroids[:, i]
    #     face_normal = unified_mesh.normals[:, i]
    #     end_point = face_center + face_normal * arrow_scale

    #     GLMakie.arrows!(ax, 
    #         [face_center[1]], [face_center[2]], [face_center[3]],
    #         [face_normal[1]], [face_normal[2]], [face_normal[3]],
    #         color = :yellow, linewidth = normal_linewidth * 1.5,
    #         arrowsize = 0.15)
    # end
    
    # GLMakie.cam3d!(ax)
    # display(f)
    # Create new triangles from midpoints of original triangles
  


    # # Create a new array to accumulate vertex normals
    # vertex_normals = zeros(Float64, 3, size(unified_mesh.vertices, 2))

    # # Iterate through faces and accumulate normals
    # for (i, face) in enumerate(eachcol(unified_mesh.faces))
    #     for vertex_index in face
    #         vertex_normals[:, vertex_index] .+= unified_mesh.normals[:, i]
    #     end
    # end

    # # Normalize the accumulated normals
    # for i in axes(vertex_normals, 2)
    #     vertex_normals[:, i] = normalize(vertex_normals[:, i])
    # end
    # arrow_scale = 0.25
    # # Create new triangles from vertex normals
    # front_triangles = zeros(Float64, 3, 3, size(unified_mesh.faces, 2))
    # back_triangles = zeros(Float64, 3, 3, size(unified_mesh.faces, 2))
    # for (i, face) in enumerate(eachcol(unified_mesh.faces))
    #     for j in 1:3
    #         vertex = unified_mesh.vertices[:, face[j]]
    #         vertex_normal = vertex_normals[:, face[j]]
    #         front_triangles[:, j, i] = vertex + vertex_normal * arrow_scale
    #         back_triangles[:, j, i] = vertex - vertex_normal * arrow_scale
    #     end
    # end

    # # After creating midpoint_triangles, add the following code:

    # Create new triangles from midpoint triangles and back triangles
    num_original_triangles = size(unified_mesh.triangles, 3)
    # midpoint_back_triangles = zeros(Float64, 3, 3, num_original_triangles * 3)
    # midpoint_front_triangles = zeros(Float64, 3, 3, num_original_triangles * 3)
    # front_subdivided_triangles = zeros(Float64, 3, 3, num_original_triangles * 3)
    # back_subdivided_triangles = zeros(Float64, 3, 3, num_original_triangles * 3)
    # triangles_from_quads1 = zeros(Float64, 3, 3, num_original_triangles * 4)
    tetrahedrons = Vector{Tetrahedron}(undef,23)
    tetrahedrons_group = Vector{TetrahedronGroup}(undef,num_original_triangles)
    for i in 1:num_original_triangles
        midpoint_triangle = view(midpoint_triangles, :, :, i)
        back_triangle = view(back_triangles, :, :, i)
        front_triangle = view(front_triangles, :, :, i)
        front_midpoint_triangle = view(front_midpoint_triangles, :, :, i)
        back_midpoint_triangle = view(back_midpoint_triangles, :, :, i)
        #  front_centroid = mean(front_triangle, dims=2)[:, 1]
        # Calculate centroids of the back and front triangles
        back_centroid = mean(back_triangle, dims=2)[:, 1]
        front_centroid = mean(front_triangle, dims=2)[:, 1]

        # Create three new triangles for both back and front
        # triangles_from_quads,midpoint_back_triangles,midpoint_front_triangles,front_subdivided_triangles,back_subdivided_triangles
        tetrahedrons[1] = Tetrahedron(midpoint_triangle[:, 1], midpoint_triangle[:, 2], midpoint_triangle[:, 3], back_centroid)
        tetrahedrons[2] = Tetrahedron(midpoint_triangle[:, 1], midpoint_triangle[:, 2], midpoint_triangle[:, 3], front_centroid)
        for j in 1:3
            # midpoint_back_triangles[:, :, (i-1)*3 + j] = [midpoint_triangle[:, j] midpoint_triangle[:, mod1(j+1, 3)] back_centroid]
            # midpoint_front_triangles[:, :, (i-1)*3 + j] = [midpoint_triangle[:, j] midpoint_triangle[:, mod1(j+1, 3)] front_centroid]
            # front_subdivided_triangles[:, :, (i-1)*3 + j] = [front_triangle[:, j] front_triangle[:, mod1(j+1, 3)] front_centroid]
            # back_subdivided_triangles[:, :, (i-1)*3 + j] = [back_triangle[:, j] back_triangle[:, mod1(j+1, 3)] back_centroid]
            

            tetrahedrons[(j-1)*7 + 3] = Tetrahedron(midpoint_triangle[:, j], midpoint_triangle[:, mod1(j+2, 3)], front_triangle[:, j], back_triangle[:, j])
            tetrahedrons[(j-1)*7 + 4] = Tetrahedron(midpoint_triangle[:, j], back_centroid,back_triangle[:, j],back_midpoint_triangle[:, j])
            tetrahedrons[(j-1)*7 + 5] = Tetrahedron(midpoint_triangle[:, j], back_centroid,back_midpoint_triangle[:, j],back_triangle[:, mod1(j+1, 3)])
            # tetrahedrons[(i-1)*12 + (j-1)*2 + 3] = Tetrahedron(midpoint_triangle[:, j], midpoint_triangle[:, mod1(j+2, 3)], front_triangle[:, j], back_triangle[:, j])
            tetrahedrons[(j-1)*7 + 6] = Tetrahedron(midpoint_triangle[:, j], front_centroid,front_triangle[:, j],front_midpoint_triangle[:, j])
            tetrahedrons[(j-1)*7 + 7] = Tetrahedron(midpoint_triangle[:, j], front_centroid,front_midpoint_triangle[:, j],front_triangle[:, mod1(j+1, 3)])
            tetrahedrons[(j-1)*7 + 8] = Tetrahedron(front_centroid, midpoint_triangle[:, j], front_triangle[:, j], midpoint_triangle[:,  mod1(j+2, 3)])
            tetrahedrons[(j-1)*7 + 9] = Tetrahedron(back_centroid, midpoint_triangle[:, j], back_triangle[:, j], midpoint_triangle[:,  mod1(j+2, 3)])
        end
        tetrahedrons_group[i] = TetrahedronGroup(tetrahedrons,unified_mesh.triangles[:,:,i])
    end




    # # Plotting


    
    # arrow_scale = 0.1
    

    

    # Plot tetrahedrons
    # for k in 1:4
    # for (k,tetrahedron_group) in enumerate(tetrahedrons_group)
    #     @show k,tetrahedron_group
    #     @show typeof(tetrahedron_group)
    #     for (m,tetrahedron) in enumerate(tetrahedron_group.tetrahedra)
    #         vertices = get_vertices(tetrahedron)# .+ (rand()-0.5)/10
    #         @show vertices

    #         # if m == 6 || m == 7 || m == 11 || m == 12 || m == 16 || m == 17
    #             # if m == 9 || m == 2 || m == 10 || m == 11 || m == 1 || m == 8
    #             for (i, j) in [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    #             v1 = vertices[i]
    #             v2 = vertices[j]
    #             # GLMakie.lines!(ax4[m < 3 ? 1 : m == 7 || m == 8 ? 2 : m == 5 || m == 4 || m == 10 || m == 11 ? 4 : 3], [v1[1], v2[1]], [v1[2], v2[2]], [v1[3], v2[3]], 
    #             # color = tet_color[mod(m, 4)+1])
    #             GLMakie.lines!(ax, [v1[1], v2[1]], [v1[2], v2[2]], [v1[3], v2[3]], 
    #                 color = tet_color[mod(m, 6)+1],linestyle = linestyle[mod(m, 4)+1])
    #             # GLMakie.text!(ax, "$(m)", position=(centroid[1], centroid[2], centroid[3]),
    #             # color=rainbow[mod(m, 17)+1], align=(:center, :center), fontsize=10)
    #         end
    #         centroid = mean(vertices)
    #         # GLMakie.text!(ax4[m < 3 ? 1 : m == 7 || m == 8 ? 2 : m == 5 || m == 4 || m == 10 || m == 11 ? 4 : 3], "$(m)", position=(centroid[1], centroid[2], centroid[3]),
    #                 # color=tet_color[mod(m, 4)+1], align=(:center, :center), fontsize=18)
    #         GLMakie.text!(ax, "$(m)", position=(centroid[1], centroid[2], centroid[3]),
    #                 color=tet_color[mod(m, 6)+1], align=(:center, :center), fontsize=18)
    #         # end
    #     end
    #     if k == 1
    #         break
    #     end
    #     # Calculate and plot the centroid
        
    #     # GLMakie.scatter!(ax, [centroid[1]], [centroid[2]], [centroid[3]], 
    #     #                  color=rainbow[mod(k, 17)+1], markersize=10)
        
    #     # Label the centroid with 'k'
        
        
    #     # @show rainbow[mod(k, 17)+1]
    
    #     # v1 = tetrahedron_group.base_triangle[:, 1]
    #     # v2 = tetrahedron_group.base_triangle[:, 2]
    #     # v3 = tetrahedron_group.base_triangle[:, 3]
    #     # # @show v1, v2, v3
    #     # GLMakie.lines!(ax, [v1[1], v2[1], v3[1], v1[1]], [v1[2], v2[2], v3[2], v1[2]], [v1[3], v2[3], v3[3], v1[3]], color = rainbow[mod(k, 17)+1] )
    # end
      # Plot tetrahedrons
      for (k,tetrahedron_group) in enumerate(tetrahedrons_group)
        @show k,tetrahedron_group
        @show typeof(tetrahedron_group)

        # Create array for neighboring tetrahedra and unmatched faces
        num_tetrahedra = length(tetrahedron_group.tetrahedra)
        neighbors = fill("", (4, num_tetrahedra))
        unmatched_faces = []

        # Identify neighbors and unmatched faces
        for (i, tetra1) in enumerate(tetrahedron_group.tetrahedra)
            for (j, face1) in enumerate(get_faces(tetra1))
                neighbor_found = false
                for (m, tetra2) in enumerate(tetrahedron_group.tetrahedra)
                    if i != m
                        for (n, face2) in enumerate(get_faces(tetra2))
                            if are_faces_equal(face1, face2)
                                neighbors[j, i] = string(m)
                                neighbor_found = true
                                break
                            end
                        end
                    end
                    if neighbor_found
                        break
                    end
                end
                if !neighbor_found
                    neighbors[j, i] = "Unmatched"
                    push!(unmatched_faces, (i, j, face1))
                end
            end
        end

        # Reconstruct the triangular prism from unmatched faces
        prism_faces = reconstruct_prism(unmatched_faces)


        @show neighbors
        # Display the reconstructed prism faces
        println("\nReconstructed Prism Faces:")
        for (i, face) in enumerate(prism_faces)
            println("Face $i: Tetrahedron $(face[1]), Face $(face[2])")
        end
        #  # Create a header for the table
        #  header = ["Face" * string(i) for i in 1:4]
        
        #  # Create row names (tetrahedron numbers)
        #  row_names = ["Tetra" * string(i) for i in 1:num_tetrahedra]
 
        #  # Display the neighbors table
        #  println("\nNeighbors and Exterior Surfaces for Tetrahedron Group $k:")
        #  pretty_table(neighbors', header, row_names=row_names, alignment=:c)
    # Calculate the centroid of the tetrahedron_group
    group_centroid = zeros(3)
    for tetrahedron in tetrahedron_group.tetrahedra
        group_centroid .+= calculate_centroid(tetrahedron)
    end
    group_centroid ./= length(tetrahedron_group.tetrahedra)
    if k != 9 
        continue
    end
        for (m,tetrahedron) in enumerate(tetrahedron_group.tetrahedra)
            
            vertices = get_vertices(tetrahedron)
            tetra_centroid = calculate_centroid(tetrahedron)
            
            # Move vertices away from group_centroid based on tetrahedron centroid
            move_vector = 0.15 .* normalize(tetra_centroid .- group_centroid)
            moved_vertices = [v .+ move_vector for v in vertices]
            
            for (i, j) in [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
                v1 = moved_vertices[i]
                v2 = moved_vertices[j]
                GLMakie.lines!(ax, [v1[1], v2[1]], [v1[2], v2[2]], [v1[3], v2[3]], 
                    color = tet_color[mod(m, 6)+1])
            end
            centroid = mean(moved_vertices)
            GLMakie.text!(ax, "$(m)", position=(centroid[1], centroid[2], centroid[3]),
                    color=tet_color[mod(m, 6)+1], align=(:center, :center), fontsize=18)
        end

        # if k == 10
        #     break
        # end
    end
    # end
    display(f)
end

main()