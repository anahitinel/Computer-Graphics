#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>
static std::default_random_engine engine(10); // random seed = 10
static std::uniform_real_distribution<double> uniform(0, 1);

#define M_PI 3.14159265358979323846264338327950288

double sqr(double x) {
    return x * x;
}

// Add these matrix transform functions at the beginning of the file, after the Vector class

// Matrix 3x3 class for rotations
class Matrix {
public:
    double data[3][3];
    
    Matrix() {
        // Identity matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = (i == j) ? 1 : 0;
            }
        }
    }
    
    static Matrix rotateY(double angle) {
        Matrix m;
        m.data[0][0] = cos(angle);
        m.data[0][2] = sin(angle);
        m.data[2][0] = -sin(angle);
        m.data[2][2] = cos(angle);
        return m;
    }
    
    static Matrix rotateX(double angle) {
        Matrix m;
        m.data[1][1] = cos(angle);
        m.data[1][2] = -sin(angle);
        m.data[2][1] = sin(angle);
        m.data[2][2] = cos(angle);
        return m;
    }
};

void boxMuller(double stdev, double &x, double &y) {
    double r1 = uniform(engine);
    double r2 = uniform(engine);
    x= sqrt(-2 * log(r1)) * cos(2 * M_PI*r2) *stdev;
    y= sqrt(-2 * log(r1)) * cos(2 * M_PI*r2) *stdev;
}

struct Vertex {
    double x, y, z;
};

struct TexCoord {
    double u, v;
};

struct Normal {
    double x, y, z;
};

struct Face {
    int v1, v2, v3;          // Vertex indices
    int vt1, vt2, vt3;       // Texture coordinate indices
    int vn1, vn2, vn3;       // Normal indices
    int material_id;         // Material index for the face
};

class Vector {
public:
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}
Vector operator-(const Vector& a) {
    return Vector(-a[0], -a[1], -a[2]);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

// Apply matrix to vector
Vector operator*(const Matrix& m, const Vector& v) {
    return Vector(
        m.data[0][0] * v[0] + m.data[0][1] * v[1] + m.data[0][2] * v[2],
        m.data[1][0] * v[0] + m.data[1][1] * v[1] + m.data[1][2] * v[2],
        m.data[2][0] * v[0] + m.data[2][1] * v[1] + m.data[2][2] * v[2]
    );
}

Vector random_cos(const Vector& N) {
    // Create a coordinate system around N
    Vector T1;
    if (std::abs(N[0]) < std::abs(N[1]))
        T1 = Vector(0, N[2], -N[1]);
    else
        T1 = Vector(N[2], 0, -N[0]);
    T1.normalize();
    Vector T2 = cross(N, T1);
    
    // Generate random angles for cosine-weighted sampling
    double r1 = uniform(engine);
    double r2 = uniform(engine);
    double phi = 2 * M_PI * r1;
    double theta = acos(sqrt(r2));
    
    // Convert to Cartesian coordinates
    double x = sin(theta) * cos(phi);
    double y = sin(theta) * sin(phi);
    double z = cos(theta);
    
    // Transform to world space
    return x * T1 + y * T2 + z * N;
}

struct Material {
    std::string name;
    Vector albedo;
    std::string diffuse_map; // Path to diffuse texture
    int texture_id;          // Index to the loaded texture
};

struct Texture {
    int width, height, channels;
    unsigned char* data;
};

struct Intersection {
  bool intersected = false;
  bool reflective = false;
  bool is_light = false;
  double refractive_index = 1.;
  double t;
  Vector sphere_center;
  Vector P;
  Vector N;
  Vector albedo;
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<TexCoord> texcoords;
    std::vector<Normal> normals;
    std::vector<Face> faces;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    // Function to load materials from MTL file
    void loadMTL(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open material file: " << filename << std::endl;
            return;
        }

        Material current_material;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string token;
            ss >> token;

            if (token == "newmtl") {
                // Save the previous material if it exists
                if (!current_material.name.empty()) {
                    materials.push_back(current_material);
                }
                // Start a new material
                current_material = Material();
                ss >> current_material.name;
                current_material.texture_id = -1; // No texture by default
            } 
            else if (token == "Kd") {
                // Diffuse color
                double r, g, b;
                ss >> r >> g >> b;
                current_material.albedo = Vector(r, g, b);
            }
            else if (token == "map_Kd") {
                // Diffuse texture map
                ss >> current_material.diffuse_map;
            }
        }

        // Don't forget to add the last material
        if (!current_material.name.empty()) {
            materials.push_back(current_material);
        }
        
        // Load textures referenced in materials
        for (size_t i = 0; i < materials.size(); i++) {
            if (!materials[i].diffuse_map.empty()) {
                Texture tex;
                tex.data = stbi_load(materials[i].diffuse_map.c_str(), &tex.width, &tex.height, &tex.channels, 3);
                if (tex.data) {
                    textures.push_back(tex);
                    materials[i].texture_id = textures.size() - 1;
                } else {
                    std::cerr << "Failed to load texture: " << materials[i].diffuse_map << std::endl;
                }
            }
        }
    }

    // Function to load an OBJ file
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        // Get the directory of the OBJ file for loading materials
        std::string directory = filename.substr(0, filename.find_last_of("/\\") + 1);
    
        // Create rotation matrix for 45 degrees around Y axis
        double angle = -45.0 * M_PI / 180.0;
        Matrix rotY = Matrix::rotateY(angle);
        
        int current_material_id = 0;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string type;
            ss >> type;
            
            if (type == "v") {
                // Read vertices
                Vertex v;
                ss >> v.x >> v.y >> v.z;
                
                // Scale the model
                v.x *= 0.6;
                v.y *= 0.6;
                v.z *= 0.6;
                
                // Apply rotation around Y axis (45 degrees)
                Vector vrot = rotY * Vector(v.x, v.y, v.z);
                v.x = vrot[0];
                v.y = vrot[1];
                v.z = vrot[2];
                
                // Translate to place it on the ground
                v.y -= 10.0;
                
                vertices.push_back(v);
            }
            else if (type == "vt") {
                // Read texture coordinates
                TexCoord vt;
                ss >> vt.u >> vt.v;
                texcoords.push_back(vt);
            }
            else if (type == "vn") {
                // Read normals
                Normal vn;
                ss >> vn.x >> vn.y >> vn.z;
                normals.push_back(vn);
            }
            else if (type == "f") {
                // Read faces (vertices/texcoords/normals indices)
                Face f;
                f.material_id = current_material_id;
                
                std::string v1, v2, v3;
                ss >> v1 >> v2 >> v3;
                
                // Parse vertex/texcoord/normal indices
                // Format: v/vt/vn
                auto parseVertex = [](const std::string& str, int& v, int& vt, int& vn) {
                    std::stringstream ss(str);
                    std::string token;
                    
                    // Get vertex index
                    std::getline(ss, token, '/');
                    v = token.empty() ? 0 : std::stoi(token);
                    
                    // Get texture coordinate index
                    std::getline(ss, token, '/');
                    vt = token.empty() ? 0 : std::stoi(token);
                    
                    // Get normal index
                    std::getline(ss, token, '/');
                    vn = token.empty() ? 0 : std::stoi(token);
                };
                
                parseVertex(v1, f.v1, f.vt1, f.vn1);
                parseVertex(v2, f.v2, f.vt2, f.vn2);
                parseVertex(v3, f.v3, f.vt3, f.vn3);
                
                faces.push_back(f);
            }
            else if (type == "mtllib") {
                // Material library reference
                std::string mtl_file;
                ss >> mtl_file;
                loadMTL(directory + mtl_file);
            }
            else if (type == "usemtl") {
                // Use material
                std::string material_name;
                ss >> material_name;
                
                // Find the material ID by name
                for (size_t i = 0; i < materials.size(); i++) {
                    if (materials[i].name == material_name) {
                        current_material_id = i;
                        break;
                    }
                }
            }
        }
        
        // If no materials were loaded but we know we need cat_diff.png,
        // load it manually
        if (materials.empty()) {
            Material default_mat;
            default_mat.name = "default";
            default_mat.albedo = Vector(0.8, 0.6, 0.2);
            default_mat.diffuse_map = "cat_diff.png";
            materials.push_back(default_mat);
            
            // Load the texture
            Texture tex;
            tex.data = stbi_load(default_mat.diffuse_map.c_str(), &tex.width, &tex.height, &tex.channels, 3);
            if (tex.data) {
                textures.push_back(tex);
                materials[0].texture_id = 0;
            } else {
                std::cerr << "Failed to load texture: " << default_mat.diffuse_map << std::endl;
            }
        }
    }
    
    ~Mesh() {
        // Free loaded textures
        for (auto& tex : textures) {
            if (tex.data) {
                stbi_image_free(tex.data);
            }
        }
    }
};

class Ray {
public:
    Ray(const Vector& O, const Vector u) : O(O), u(u) {};
    Vector O, u;
};

class Object {
public:
    virtual bool intersect(const Ray& r, Vector& P, Vector& N, double& t, Vector& albedo) = 0;
    virtual bool isMirror() const = 0;     // Added to check if object is a mirror
    virtual bool isTransparent() const = 0; // Added to check if object is transparent
    virtual ~Object() {}  // Add virtual destructor for proper cleanup
};

class Sphere: public Object {
public:
    Sphere(const Vector& C, double R, const Vector& albedo, double reflectivity, bool isMirror, bool isTransparent, bool flip_normal)
    : C(C), R(R), albedo(albedo), reflectivity(reflectivity), mirror(isMirror), transparent(isTransparent), flip_normal(flip_normal) {}

    bool intersect(const Ray& r, Vector &P, Vector &N, double& t, Vector& color) override {
        double delta = (dot(r.u, r.O - C)) * (dot(r.u, r.O - C)) - ((r.O - C).norm2() - R * R);
        if (delta < 0) return false;
        double x = dot(r.u, C - r.O);
        double t1 = x - sqrt(delta);
        double t2 = x + sqrt(delta);
        if (t2 < 0) return false;
        if (t1 > 0) {
            t = t1;
        } else {
            t = t2;
        }
        P = r.O + t * r.u;
        N = P - C;
        if (flip_normal){
            N = -1 * N;
        }
        N.normalize();
        color = albedo;
        return true;
    }
    
    bool isMirror() const override {
        return mirror;
    }
    
    bool isTransparent() const override {
        return transparent;
    }
    
    Vector C;
    double R;
    Vector albedo;
    double reflectivity;
    bool mirror;
    bool transparent;
    bool flip_normal;
};

class BoundingBox {
public:
    Vector bmin, bmax;
    
    BoundingBox() {
        // Initialize with extreme values
        bmin = Vector(std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max());
        bmax = Vector(-std::numeric_limits<double>::max(),
                      -std::numeric_limits<double>::max(),
                      -std::numeric_limits<double>::max());
    }
    
    // Check if ray intersects with this bounding box
    bool intersect(const Ray& r) const {
        double tx1 = (bmin[0] - r.O[0]) / r.u[0];
        double tx2 = (bmax[0] - r.O[0]) / r.u[0];
        
        double ty1 = (bmin[1] - r.O[1]) / r.u[1];
        double ty2 = (bmax[1] - r.O[1]) / r.u[1];
        
        double tz1 = (bmin[2] - r.O[2]) / r.u[2];
        double tz2 = (bmax[2] - r.O[2]) / r.u[2];
        
        double tmin = std::min(tx1, tx2);
        double tmax = std::max(tx1, tx2);
        
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
        
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));
        
        return tmax >= tmin && tmax > 0;
    }
};

class MeshObject : public Object {
public:
    Mesh mesh;
    BoundingBox bbox;
    bool mirror;
    bool transparent;

    MeshObject(const std::string& filename, bool isMirror = false, bool isTransparent = false) 
        : mirror(isMirror), transparent(isTransparent) {
        mesh.load(filename);
        computeBoundingBox();
    }
    
    void computeBoundingBox() {
        for (const auto& v : mesh.vertices) {
            bbox.bmin[0] = std::min(bbox.bmin[0], v.x);
            bbox.bmin[1] = std::min(bbox.bmin[1], v.y);
            bbox.bmin[2] = std::min(bbox.bmin[2], v.z);
            
            bbox.bmax[0] = std::max(bbox.bmax[0], v.x);
            bbox.bmax[1] = std::max(bbox.bmax[1], v.y);
            bbox.bmax[2] = std::max(bbox.bmax[2], v.z);
        }
    }

    bool intersect(const Ray& r, Vector& P, Vector& N, double& t, Vector& color) override {
        // First check if ray intersects the bounding box
        if (!bbox.intersect(r)) {
            return false;
        }
        
        bool hit = false;
        t = std::numeric_limits<double>::max();
        int hit_face_idx = -1;

        // Check intersection with each face of the mesh
        for (size_t i = 0; i < mesh.faces.size(); i++) {
            const Face& f = mesh.faces[i];
            Vertex v1 = mesh.vertices[f.v1 - 1];  // OBJ indices are 1-based
            Vertex v2 = mesh.vertices[f.v2 - 1];
            Vertex v3 = mesh.vertices[f.v3 - 1];

            Vector A(v1.x, v1.y, v1.z);
            Vector B(v2.x, v2.y, v2.z);
            Vector C(v3.x, v3.y, v3.z);

            // Ray-triangle intersection test (Möller–Trumbore algorithm)
            Vector e1 = B - A;
            Vector e2 = C - A;
            Vector h = cross(r.u, e2);
            double a = dot(e1, h);
            
            // Check if ray is parallel to triangle
            if (a > -1e-7 && a < 1e-7)
                continue;

            double factor = 1.0 / a;
            Vector s = r.O - A;
            double u = factor * dot(s, h);
            
            // Check if intersection point is outside triangle
            if (u < 0.0 || u > 1.0)
                continue;

            Vector q = cross(s, e1);
            double v = factor * dot(r.u, q);
            
            // Check if intersection point is outside triangle
            if (v < 0.0 || u + v > 1.0)
                continue;

            double t_temp = factor * dot(e2, q);
            
            // Check if intersection point is behind the ray origin
            if (t_temp > 1e-7 && t_temp < t) {
                t = t_temp;
                P = r.O + r.u * t;
                
                // Store geometry normal from triangle
                N = cross(e1, e2);
                N.normalize();
                
                // Calculate barycentric coordinates for texture mapping
                double alpha = 1.0 - u - v;  // First barycentric coordinate
                double beta = u;             // Second barycentric coordinate
                double gamma = v;            // Third barycentric coordinate
                
                hit = true;
                hit_face_idx = i;
                
                // If this face has UV coordinates and the mesh has a texture, use it
                const Face& face = mesh.faces[hit_face_idx];
                int material_id = face.material_id;
                
                if (material_id >= 0 && material_id < mesh.materials.size()) {
                    int texture_id = mesh.materials[material_id].texture_id;
                    
                    // If there's a texture and valid texture coordinates
                    if (texture_id >= 0 && texture_id < mesh.textures.size() && 
                        face.vt1 > 0 && face.vt2 > 0 && face.vt3 > 0 &&
                        face.vt1 <= mesh.texcoords.size() && 
                        face.vt2 <= mesh.texcoords.size() && 
                        face.vt3 <= mesh.texcoords.size()) {
                        
                        // Get texture coordinates for each vertex of the triangle
                        TexCoord tc1 = mesh.texcoords[face.vt1 - 1];
                        TexCoord tc2 = mesh.texcoords[face.vt2 - 1];
                        TexCoord tc3 = mesh.texcoords[face.vt3 - 1];
                        
                        // Interpolate texture coordinates using barycentric coordinates
                        double u_interp = alpha * tc1.u + beta * tc2.u + gamma * tc3.u;
                        double v_interp = alpha * tc1.v + beta * tc2.v + gamma * tc3.v;
                        
                        // Ensure texture coordinates are in [0,1] range (taking fractional part)
                        u_interp = u_interp - floor(u_interp);
                        v_interp = v_interp - floor(v_interp);
                        
                        // Get texture width and height
                        int tex_width = mesh.textures[texture_id].width;
                        int tex_height = mesh.textures[texture_id].height;
                        
                        // Convert texture coordinates to pixel coordinates
                        int x = static_cast<int>(u_interp * tex_width);
                        int y = static_cast<int>((1.0 - v_interp) * tex_height);  // Flip Y to match texture coordinates
                        
                        // Clamp coordinates to valid range
                        x = std::max(0, std::min(x, tex_width - 1));
                        y = std::max(0, std::min(y, tex_height - 1));
                        
                        // Get color from texture
                        unsigned char* pixel = &mesh.textures[texture_id].data[(y * tex_width + x) * 3];
                        color = Vector(
                            pixel[0] / 255.0,
                            pixel[1] / 255.0,
                            pixel[2] / 255.0
                        );
                        
                        // Apply gamma correction to texture color
                        color = Vector(
                            pow(color[0], 2.2),
                            pow(color[1], 2.2),
                            pow(color[2], 2.2)
                        );
                    } else {
                        // Fallback to material color
                        color = mesh.materials[material_id].albedo;
                    }
                } else {
                    // Default color if no material
                    color = Vector(0.8, 0.6, 0.2);
                }
                
                // If normals are available, interpolate them as well
                if (face.vn1 > 0 && face.vn2 > 0 && face.vn3 > 0 &&
                    face.vn1 <= mesh.normals.size() && 
                    face.vn2 <= mesh.normals.size() && 
                    face.vn3 <= mesh.normals.size()) {
                    
                    Normal n1 = mesh.normals[face.vn1 - 1];
                    Normal n2 = mesh.normals[face.vn2 - 1];
                    Normal n3 = mesh.normals[face.vn3 - 1];
                    
                    // Interpolate normals using barycentric coordinates
                    Vector interp_normal(
                        alpha * n1.x + beta * n2.x + gamma * n3.x,
                        alpha * n1.y + beta * n2.y + gamma * n3.y,
                        alpha * n1.z + beta * n2.z + gamma * n3.z
                    );
                    
                    interp_normal.normalize();
                    N = interp_normal;
                }
            }
        }
        return hit;
    }
    
    bool isMirror() const override {
        return mirror;
    }
    
    bool isTransparent() const override {
        return transparent;
    }
};

class Scene {
public:
    Scene() {}
    
    // Destructor to clean up dynamically allocated objects
    ~Scene() {
        for (auto obj : objects) {
            delete obj;
        }
    }

    // Store Object* pointers
    std::vector<Object*> objects;

    void add_sphere(Sphere* sph) {
        objects.push_back(sph);
    }

    void add_obj(Object* obj) {
        objects.push_back(obj);
    }

    bool intersect(const Ray& r, Vector& P, Vector& N, double& t, int& object_id, Vector& albedo) {
        bool result = false;
        t = std::numeric_limits<double>::max();
        for (int i = 0; i < objects.size(); i++) {
            Vector localN, localP;
            double localt;
            Vector localAlbedo;
            if (objects[i]->intersect(r, localP, localN, localt, localAlbedo)) {
                result = true;
                if (localt < t) {
                    t = localt;
                    P = localP;
                    N = localN;
                    object_id = i;
                    albedo = localAlbedo;
                }
            }
        }
        return result;
    }

    bool ShadowRay(const Vector& P, const Vector& L, double d2) {
        Ray shadow_ray(P + L * 1e-4, L);
        double t;
        Vector N;
        Vector localP = P;
        int object_id;
        Vector albedo;
        if (intersect(shadow_ray, localP, N, t, object_id, albedo)) {
            return t * t < d2;
        }
        return false;
    }

    Vector getColor(const Ray& r, int ray_depth) {
        if (ray_depth < 0) return Vector(0., 0., 0.);
        Vector P, N;
        double t;
        Vector color;
        int object_id;
        Vector albedo;
        double eps = 1E-8;

        if (intersect(r, P, N, t, object_id, albedo)) {
            Object* hitObject = objects[object_id];
            
            // Basic lighting implementation
            Vector L = light_position - P;
            double d2 = L.norm2();
            L.normalize();
            
            // Simple diffuse lighting
            double diffuse = std::max(0.0, dot(N, L));
            
            // Check for shadows
            bool is_in_shadow = ShadowRay(P, L, d2);
            
            // Calculate direct illumination
            if (!is_in_shadow) {
                color = albedo * diffuse * light_intensity / (4 * M_PI * d2);
            } else {
                // In shadow - use ambient lighting
                color = albedo * 0.1; // Ambient factor
            }
            
            // Add specular highlight for non-transparent objects
            if (!hitObject->isTransparent()) {
                Vector view = -1.0 * r.u;  // Direction to viewer
                Vector reflect = 2 * dot(N, L) * N - L;  // Reflection direction
                reflect.normalize();
                double spec = std::max(0.0, dot(view, reflect));
                double specular = 0.4 * pow(spec, 50);  // Specular intensity and shininess
                color = color + Vector(specular, specular, specular);
            }

            // Add indirect lighting if we have ray depth left
            if (ray_depth > 0) {
                // Generate random direction in hemisphere for indirect lighting
                Vector raydir = random_cos(N);
                Ray randomRay(P + eps * N, raydir);
                
                // Recursive call to get indirect lighting
                Vector indirect_color = getColor(randomRay, ray_depth - 1);
                
                // Compute contribution of indirect lighting
                // Factor of 0.5 to control indirect light intensity
                // Apply component-wise multiplication of albedo and indirect_color
                Vector scaled_indirect;
                scaled_indirect[0] = albedo[0] * indirect_color[0] * 0.5;
                scaled_indirect[1] = albedo[1] * indirect_color[1] * 0.5;
                scaled_indirect[2] = albedo[2] * indirect_color[2] * 0.5;
                color = color + scaled_indirect;
            }
            
            return color;
        }
        
        // Background color (black)
        return Vector(0., 0., 0.);
    }
    Vector light_position;
    double light_intensity;
};

int main() {
    int W = 512;
    int H = 512;

    double camera_angle = -10.0 * M_PI / 180.0;
    Matrix cameraRotX = Matrix::rotateX(camera_angle);

    Vector camera_origin(10, 10, 45);
    double fov = 60 * M_PI / 180.0;

    Sphere Smirror(Vector(20.,0.,10.), 10, Vector(0.5, 0.5, 0.5), 1., true, false, false);

    // Create the scene
    Scene scene;
    scene.light_position = Vector(-10, 20, 40);
    scene.light_intensity = 1E7;

    scene.add_sphere(&Smirror);

    // Add walls
    scene.add_sphere(new Sphere(Vector(0, -1000, 0), 990, Vector(0.16, 0.8, 0.5), 0., false, false, false)); // bottom wall, light blue
    scene.add_sphere(new Sphere(Vector(0, 0, 1000), 940, Vector(1.0, 0, 0), 0., false, false, false)); // top wall, red
    scene.add_sphere(new Sphere(Vector(0, 0, -1000), 940, Vector(0, 1.0, 0), 0., false, false, false)); // back wall, green
    scene.add_sphere(new Sphere(Vector(0, 1000, 0), 940, Vector(0, 0, 1.0), 0., false, false, false)); // front wall, blue
    scene.add_sphere(new Sphere(Vector(1000, 0, 0), 940, Vector(1.0, 1.0, 0), 0., false, false, false)); // right wall, yellow
    scene.add_sphere(new Sphere(Vector(-1000, 0, 0), 940, Vector(0.5, 0, 1.0), 0., false, false, false)); // left wall, purple

    // Load and add the cat mesh
    MeshObject* catMesh = new MeshObject("cat.obj", false, false);
    scene.add_obj(catMesh);

    // Ray tracing loop with antialiasing
    std::vector<unsigned char> image(W * H * 3, 0);
    
    // Number of samples per pixel for antialiasing
    int n_rays = 32;
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            // Accumulate color samples
            Vector accumulated_color(0, 0, 0);
            
            // Cast multiple rays per pixel with slight offsets for antialiasing
            for (int k = 0; k < n_rays; k++) {
                // Generate random offsets
                double x, y;
                boxMuller(0.5, x, y);
                
                // Calculate ray direction with offset for antialiasing
                double d = -W / (2 * tan(fov / 2));
                Vector raw_dir(j - W / 2 + 0.5 + x, H / 2 - i + 0.5 + y, d);
                
                // Apply camera rotation
                Vector r_dir = cameraRotX * raw_dir;
                r_dir.normalize();
                
                Ray r(camera_origin, r_dir);
                accumulated_color = accumulated_color + scene.getColor(r, 5);
            }
            
            // Average the accumulated color
            Vector color = accumulated_color / n_rays;

            // Gamma correction
            double gamma = 2.2;
            image[(i * W + j) * 3 + 0] = std::min(255., 255. * pow(color[0]/255, 1. / gamma));
            image[(i * W + j) * 3 + 1] = std::min(255., 255. * pow(color[1]/255, 1. / gamma));
            image[(i * W + j) * 3 + 2] = std::min(255., 255. * pow(color[2]/255, 1. / gamma));
        }
    }

    // Write the image to a PNG file
    stbi_write_png("image7.png", W, H, 3, &image[0], 0);

    return 0;
}