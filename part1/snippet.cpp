#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define M_PI 3.14159265358979323846264338327950288

double sqr(double x) {
    return x * x;
}

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
    virtual bool intersect(const Ray& r, Vector &P, Vector &N, double& t) = 0;
};



class Sphere: public Object {
public:
    Sphere(const Vector& C, double R, const Vector& albedo, double reflectivity, bool isMirror, bool isTransparent, bool flip_normal)
    : C(C), R(R), albedo(albedo), reflectivity(reflectivity), isMirror(isMirror), isTransparent(isTransparent), flip_normal(flip_normal) {}

    bool intersect(const Ray& r, Vector &P, Vector &N, double& t) {
        bool result = false;
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
    Vector C;
    double R;
    Vector albedo;
    double reflectivity;
    bool isMirror;
    bool isTransparent;
    bool flip_normal;
};

class Scene {
public:
    Scene(){};

    std::vector<Sphere> objects;

    void add(const Sphere sph) {
        objects.push_back(sph);
    };
    bool intersect(const Ray& r, Vector& P, Vector& N, double& t, int& object_id) {
        bool result = false;
        t= std::numeric_limits<double>::max();
        for (int i = 0; i < objects.size(); i++) {
            Vector localN, localP;
            double localt;
            if (objects[i].intersect(r, localP, localN, localt)) {
                result=true;
                if (localt < t) {
                    result = true;
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
            Sphere& hitSphere = objects[object_id];
            if (hitSphere.isMirror) {
                Vector reflectedDir = r.u - 2 * dot(r.u, N) * N;
                Ray reflectedRay(P + 0.0001 * reflectedDir, reflectedDir);
                return getColor(reflectedRay, ray_depth - 1);
            };
            if (hitSphere.isTransparent) {
                double n1 = 1.0;             // Air
                double n2 = 1.5;             // Glass
                Vector Nn = N;
                Vector incident = r.u;

                double cosi = dot(incident, Nn);
                bool outside = cosi < 0;
                if (!outside) {
                    std::swap(n1, n2);
                    Nn = -1 * Nn;
                    cosi = dot(incident, Nn);
                }

                double eta = n1 / n2;
                double k = 1 - eta * eta * (1 - cosi * cosi);

                Vector refractedDir;
                bool totalInternalReflection = (k < 0);

                // Compute reflected direction
                Vector reflectedDir = incident - 2 * dot(incident, Nn) * Nn;
                Ray reflectedRay(P + 1e-4 * reflectedDir, reflectedDir);
                Vector reflectedColor = getColor(reflectedRay, ray_depth - 1);

                if (totalInternalReflection) {
                    return reflectedColor;  // Fully reflective at this point
                }

                // Compute refracted direction
                refractedDir = eta * (incident - cosi * Nn) - Nn * sqrt(k);
                Ray refractedRay(P + 1e-4 * refractedDir, refractedDir);
                Vector refractedColor = getColor(refractedRay, ray_depth - 1);

                // Fresnel coefficients via Schlick's approximation
                double R0 = pow((n1 - n2) / (n1 + n2), 2);
                double R = R0 + (1 - R0) * pow(1 - fabs(cosi), 5);
                double T = 1 - R;

                return reflectedColor * R + refractedColor * T;
            }


            if (hitSphere.reflectivity > 0) {
                Vector reflectedDir = r.u - 2 * dot(r.u, N) * N;
                Ray reflectedRay(P + 0.0001 * reflectedDir, reflectedDir);
                Vector reflectedColor = getColor(reflectedRay, ray_depth - 1);
                return hitSphere.albedo * (1 - hitSphere.reflectivity) + reflectedColor * hitSphere.reflectivity;
            }

            Vector lightDir = light_position - P;
            double d2 = lightDir.norm2();
            lightDir.normalize();
            Vector color = light_intensity / (4 * M_PI * d2) * objects[object_id].albedo / M_PI * std::max(0., dot(lightDir, N));
            
            Ray shadowRay(P + 0.00001 * N, lightDir);
            Vector shadowP, shadowN;
            double shadowt;
            int shadow_id;
            if (intersect(shadowRay, shadowP, shadowN, shadowt, shadow_id)) {
                if (shadowt * shadowt < d2) {
                    color = Vector(0., 0., 0.);
                }
            }
            return color;
        }
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

    // Define the reflective sphere (mirror)
    Sphere Smirror(Vector(-20.,0.,0.), 10, Vector(0.5, 0.5, 0.5), 1., true, false, false);
    Sphere Stransparent(Vector(0.,0.,0.), 10, Vector(0.5, 0.5, 0.5), 1, false, true, false);
    Sphere S3(Vector(20.,0.,0.), 10, Vector(0.5, 0.5, 0.5), 0., false, true, false);
    Sphere S4(Vector(20.,0.,0.), 9.8, Vector(0.5, 0.5, 0.5), 0., false, true, true);
    Scene scene;
    scene.light_position = Vector(-10, 20, 40);
    scene.light_intensity = 1E7;
    scene.add(Smirror);
    scene.add(Stransparent);
    scene.add(S3);
    scene.add(S4);

    // Define walls
    scene.add(Sphere(Vector(0, -1000, 0), 990, Vector(0.16, 0.8, 0.5), 0., false, false, false)); // bottom wall, light blue
    scene.add(Sphere(Vector(0, 0, 1000), 940, Vector(1.0, 0, 0), 0., false, false, false)); // top wall, red
    scene.add(Sphere(Vector(0, 0, -1000), 940, Vector(0, 1.0, 0), 0., false, false, false)); // back wall, green
    scene.add(Sphere(Vector(0, 1000, 0), 940, Vector(0, 0, 1.0), 0., false, false, false)); // front wall, blue (0.36, 0.2, 0.7)
    scene.add(Sphere(Vector(1000, 0, 0), 940, Vector(1.0, 1.0, 0), 0., false, false, false)); // right wall, yellow
    scene.add(Sphere(Vector(-1000, 0, 0), 940, Vector(0.5, 0, 1.0), 0., false, false, false)); // left wall, purple

    int object_id;
    
    std::vector<unsigned char> image(W * H * 3, 0);
    // #pragma omp parallel for schedule(dynamic, 1)
    // Ray tracing loop
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            double d = -W / (2 * tan(fov / 2));
            Vector r_dir(j - W / 2 + 0.5, H / 2 - i + 0.5, d);
            r_dir.normalize();
            Ray r(camera_origin, r_dir);
            Vector color = scene.getColor(r, 5);  // Start with depth 5

            double gamma = 2.2;
            image[(i * W + j) * 3 + 0] = std::min(255., 255. * pow(color[0]/255., 1./gamma));
            image[(i * W + j) * 3 + 1] = std::min(255., 255. * pow(color[1]/255., 1./gamma));
            image[(i * W + j) * 3 + 2] = std::min(255., 255. * pow(color[2]/255., 1./gamma));
        }
    }

    // Write the image to a PNG file
    stbi_write_png("image2.png", W, H, 3, &image[0], 0);

    return 0;
}