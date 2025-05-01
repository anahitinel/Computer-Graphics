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
#define M_PI 3.14159265358979323846264338327950288

double sqr(double x) {
    return x * x;
}

struct Vertex {
    double x, y, z;
};

struct Face {
    int v1, v2, v3;  // Indices of the vertices forming the triangle
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    // Function to load an OBJ file
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string type;
            ss >> type;
            
            if (type == "v") {
                // Read vertices
                Vertex v;
                ss >> v.x >> v.y >> v.z;
                
                // Scale and translate the model to position it properly
                v.x *= 0.6;  // Scale by 0.6 as suggested in the document
                v.y *= 0.6;
                v.z *= 0.6;
                
                v.y -= 10.0;  // Translate by (0, -10, 0) to place it on the ground
                
                vertices.push_back(v);
            } 
            else if (type == "f") {
                // Read faces (indices of vertices forming triangles)
                Face f;
                std::string v1, v2, v3;
                ss >> v1 >> v2 >> v3;
                
                // Handle the case where vertices have texture/normal data (v/vt/vn format)
                std::stringstream ss1(v1);
                std::stringstream ss2(v2);
                std::stringstream ss3(v3);
                std::string v1_index, v2_index, v3_index;
                
                std::getline(ss1, v1_index, '/');
                std::getline(ss2, v2_index, '/');
                std::getline(ss3, v3_index, '/');
                
                f.v1 = std::stoi(v1_index);
                f.v2 = std::stoi(v2_index);
                f.v3 = std::stoi(v3_index);
                
                faces.push_back(f);
            }
        }
        
        std::cout << "Loaded mesh with " << vertices.size() << " vertices and " 
                  << faces.size() << " faces." << std::endl;
    }
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


class Ray {
public:
    Ray(const Vector& O, const Vector u) : O(O), u(u) {};
    Vector O, u;
};

class Object {
public:
    virtual bool intersect(const Ray& r, Vector& P, Vector& N, double& t) = 0;
    virtual Vector getAlbedo() const = 0;  // Added to get albedo for any object
    virtual bool isMirror() const = 0;     // Added to check if object is a mirror
    virtual bool isTransparent() const = 0; // Added to check if object is transparent
    virtual ~Object() {}  // Add virtual destructor for proper cleanup
};

class Sphere: public Object {
public:
    Sphere(const Vector& C, double R, const Vector& albedo, double reflectivity, bool isMirror, bool isTransparent, bool flip_normal)
    : C(C), R(R), albedo(albedo), reflectivity(reflectivity), mirror(isMirror), transparent(isTransparent), flip_normal(flip_normal) {}

    bool intersect(const Ray& r, Vector &P, Vector &N, double& t) override {
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
        return true;
    }
    
    Vector getAlbedo() const override {
        return albedo;
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
    Vector albedo;
    BoundingBox bbox;
    bool mirror;
    bool transparent;

    MeshObject(const std::string& filename, const Vector& albedo = Vector(0.8, 0.6, 0.2), 
               bool isMirror = false, bool isTransparent = false) 
        : albedo(albedo), mirror(isMirror), transparent(isTransparent) {
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

    bool intersect(const Ray& r, Vector& P, Vector& N, double& t) override {
        // First check if ray intersects the bounding box
        if (!bbox.intersect(r)) {
            return false;
        }
        
        bool hit = false;
        t = std::numeric_limits<double>::max();

        // Check intersection with each face of the mesh
        for (const Face& f : mesh.faces) {
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
                N = cross(e1, e2);
                N.normalize();
                hit = true;
            }
        }
        return hit;
    }
    
    Vector getAlbedo() const override {
        return albedo;
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

    bool intersect(const Ray& r, Vector& P, Vector& N, double& t, int& object_id) {
        bool result = false;
        t = std::numeric_limits<double>::max();
        for (int i = 0; i < objects.size(); i++) {
            Vector localN, localP;
            double localt;
            if (objects[i]->intersect(r, localP, localN, localt)) {
                result = true;
                if (localt < t) {
                    t = localt;
                    P = localP;
                    N = localN;
                    object_id = i;
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
        if (intersect(shadow_ray, localP, N, t, object_id)) {
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

        if (intersect(r, P, N, t, object_id)) {
            Object* hitObject = objects[object_id];
            Vector objectAlbedo = hitObject->getAlbedo();
            
            // Basic lighting implementation
            Vector L = light_position - P;
            double d2 = L.norm2();
            L.normalize();
            
            // Simple diffuse lighting
            double diffuse = std::max(0.0, dot(N, L));
            
            // Check for shadows
            bool is_in_shadow = ShadowRay(P, L, d2);
            
            // Calculate illumination
            if (!is_in_shadow) {
                color = objectAlbedo * diffuse * light_intensity / (4 * M_PI * d2);
            } else {
                // In shadow - use ambient lighting
                color = objectAlbedo * 0.1; // Ambient factor
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
    Vector camera_origin(0, 0, 55);
    double fov = 60 * M_PI / 180.0;

    // Create the scene
    Scene scene;
    scene.light_position = Vector(-10, 20, 40);
    scene.light_intensity = 1E7;

    // Add walls
    scene.add_sphere(new Sphere(Vector(0, -1000, 0), 990, Vector(0.16, 0.8, 0.5), 0., false, false, false)); // bottom wall, light blue
    scene.add_sphere(new Sphere(Vector(0, 0, 1000), 940, Vector(1.0, 0, 0), 0., false, false, false)); // top wall, red
    scene.add_sphere(new Sphere(Vector(0, 0, -1000), 940, Vector(0, 1.0, 0), 0., false, false, false)); // back wall, green
    scene.add_sphere(new Sphere(Vector(0, 1000, 0), 940, Vector(0, 0, 1.0), 0., false, false, false)); // front wall, blue (0.36, 0.2, 0.7)
    scene.add_sphere(new Sphere(Vector(1000, 0, 0), 940, Vector(1.0, 1.0, 0), 0., false, false, false)); // right wall, yellow
    scene.add_sphere(new Sphere(Vector(-1000, 0, 0), 940, Vector(0.5, 0, 1.0), 0., false, false, false)); // left wall, purple

    // Load and add the cat mesh
    MeshObject* catMesh = new MeshObject("cat.obj", Vector(0.8, 0.6, 0.2));  // Cat color with golden/brown tone
    scene.add_obj(catMesh);

    // Ray tracing loop
    std::vector<unsigned char> image(W * H * 3, 0);
    for (int i = 0; i < H; i++) {
        
        for (int j = 0; j < W; j++) {
            double d = -W / (2 * tan(fov / 2));
            Vector r_dir(j - W / 2 + 0.5, H / 2 - i + 0.5, d);
            r_dir.normalize();
            Ray r(camera_origin, r_dir);
            Vector color = scene.getColor(r, 5);

            // Gamma correction
            double gamma = 2.2;
            image[(i * W + j) * 3 + 0] = std::min(255., 255. * pow(color[0]/255, 1. / gamma));
            image[(i * W + j) * 3 + 1] = std::min(255., 255. * pow(color[1]/255, 1. / gamma));
            image[(i * W + j) * 3 + 2] = std::min(255., 255. * pow(color[2]/255, 1. / gamma));
        }
    }

    // Write the image to a PNG file
    stbi_write_png("image3.png", W, H, 3, &image[0], 0);

    return 0;
}