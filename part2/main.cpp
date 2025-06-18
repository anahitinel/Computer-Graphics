#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <sstream>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "codes/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "codes/stb_image.h"

#include "codes/vector.cpp"

static std::default_random_engine engine(10); 
static std::uniform_real_distribution<double> uniform(0, 1);

#define M_PI 3.14159265358979323846264338327950288
#define VOL_FLUID 0.6

class Polygon {  
public:
    std::vector<Vector> vertices;

    double area() {
        if (vertices.size() < 3) return 0;
        double s = 0;
        for (int i=0; i<vertices.size(); i++) {
            int ip = (i==vertices.size()-1) ? 0 : (i+1);
            s += vertices[i][0] * vertices[ip][1] - vertices[ip][0] * vertices[i][1];
        }
        return std::abs(s) / 2.;
    }

    Vector centroid() {
        if(vertices.size() < 3) return Vector(0, 0, 0);
        Vector c(0, 0, 0);
        for (int i=0; i<vertices.size(); i++) {
            int ip = (i==vertices.size() - 1) ? 0 : (i + 1);
            double crossP = (vertices[i][0] * vertices[ip][1] - vertices[ip][0] * vertices[i][1]);
            c = c + (vertices[i] + vertices[ip])*crossP;
        }
        double a = area();
        c = c/(6.*a);
        return c;
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;
        double s = 0;
        for(int t = 1; t < vertices.size() - 1; t++) {
            Vector c[3] = {vertices[0], vertices[t], vertices[t+1]};
            double integralT = 0;
            for (int k=0; k<3; k++) {
                for (int l = k; l<3; l++) {
                    integralT += dot(c[k] - Pi, c[l] - Pi);
                }
            }
            Vector edge1 = c[1] - c[0];
            Vector edge2 = c[2] - c[1];
            double areaT = 0.5 * std::abs(edge1[0] * edge2[1] - edge1[1] * edge2[0]);
            s+= integralT * areaT / 6.;
        }
        return s;
    }
};

class VoronoiDiagram {
public:
    int N_disk = 100;
    std::vector<Vector> unit_disk;
    std::vector<Vector> points;
    std::vector<Polygon> cells;
    std::vector<double> weights;

    VoronoiDiagram() {
        unit_disk.resize(N_disk);
        for(int i=0; i<N_disk; i++) {
            double theta = i/(double)N_disk * 2 * M_PI;
            unit_disk[i] = Vector(sin(-theta), cos(-theta), 0);
        }
    }

    Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {
        Vector N(v[1] - u[1], u[0] - v[0], 0);
        double norm = sqrt(N[0]*N[0] + N[1]*N[1]);
        if (norm != 0) N = N / norm;
        Polygon result;
        for (int i = 0; i < V.vertices.size(); i++) {
            const Vector& A = V.vertices[i];
            const Vector& B = V.vertices[(i + 1) % V.vertices.size()];
            double sideA = dot(A - u, N);
            double sideB = dot(B - u, N);
            bool insideA = sideA <= 0;
            bool insideB = sideB <= 0;
            if (insideA && insideB) {
                result.vertices.push_back(B);
            } else if (insideA && !insideB) {
                Vector dir = B - A;
                double t = sideA / (sideA - sideB);
                Vector I = A + t * dir;
                result.vertices.push_back(I);
            } else if (!insideA && insideB) {
                Vector dir = B - A;
                double t = sideA / (sideA - sideB);
                Vector I = A + t * dir;
                result.vertices.push_back(I);
                result.vertices.push_back(B);
            }
        }
        return result;
    }

    Polygon clip_by_bisector(const Polygon& V, const Vector& Pi, const Vector& Pj, double wi, double wj) {
        Vector u = Pj - Pi;
        double norm_u2 = dot(u, u);
        if (norm_u2 == 0) return V;
        Vector N = u;
        double c = 0.5 * (dot(Pj, Pj) - wj - dot(Pi, Pi) + wi);
        double offset = c / norm_u2;
        Vector M = Pi + offset * u;
        Polygon result;
        for (int i = 0; i < V.vertices.size(); ++i) {
            const Vector& A = V.vertices[i];
            const Vector& B = V.vertices[(i + 1) % V.vertices.size()];
            double sideA = dot(A - M, N);
            double sideB = dot(B - M, N);
            bool insideA = sideA <= 0;
            bool insideB = sideB <= 0;
            if (insideA && insideB) {
                result.vertices.push_back(B);
            } else if (insideA && !insideB) {
                Vector dir = B - A;
                double t = sideA / (sideA - sideB);
                Vector I = A + t * dir;
                result.vertices.push_back(I);
            } else if (!insideA && insideB) {
                Vector dir = B - A;
                double t = sideA / (sideA - sideB);
                Vector I = A + t * dir;
                result.vertices.push_back(I);
                result.vertices.push_back(B);
            }
        }
        return result;
    }

    void compute() {
        Polygon square;
        square.vertices = { Vector(0, 0), Vector(0, 1), Vector(1, 1), Vector(1, 0) };
        cells.resize(points.size());
        for (int i = 0; i < points.size(); i++) {
            Polygon V = square;
            for (int j = 0; j < points.size(); j++) {
                if (i == j) continue;
                V = clip_by_bisector(V, points[i], points[j], weights[i], weights[j]);
            }
            /*
            for (int j = 0; j < N_disk; j++) {
                double radius = sqrt(std::max(0.0, weights[i] - weights[weights.size() - 1]));
                Vector u = unit_disk[j] * radius + points[i];
                Vector v = unit_disk[(j + 1) % N_disk] * radius + points[i];
                V = clip_by_edge(V, u, v);
            } */
            cells[i] = V;
        }
    }
};

int sgn(double x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

void save_frame(std::vector<Polygon> &cells, std::string filename, int frameid = 0, int N =1000) {
    int W = 1000, H = 1000;
    std::vector<unsigned char> image(W*H * 3, 255);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < cells.size(); i++) {
 
        double bminx = 1E9, bminy = 1E9, bmaxx = -1E9, bmaxy = -1E9;
        for (int j = 0; j < cells[i].vertices.size(); j++) {
            bminx = std::min(bminx, cells[i].vertices[j][0]);
            bminy = std::min(bminy, cells[i].vertices[j][1]);
            bmaxx = std::max(bmaxx, cells[i].vertices[j][0]);
            bmaxy = std::max(bmaxy, cells[i].vertices[j][1]);
        }
        bminx = std::min(W-1., std::max(0., W * bminx));
        bminy = std::min(H-1., std::max(0., H * bminy));
        bmaxx = std::max(W-1., std::max(0., W * bmaxx));
        bmaxy = std::max(H-1., std::max(0., H * bmaxy));

        for (int y = bminy; y < bmaxy; y++) {
            for (int x = bminx; x < bmaxx; x++) {
                int prevSign = 0;
                bool isInside = true;
                double mindistEdge = 1E9;
                for (int j = 0; j < cells[i].vertices.size(); j++) {
                    double x0 = cells[i].vertices[j][0] * W;
                    double y0 = cells[i].vertices[j][1] * H;
                    double x1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][0] * W;
                    double y1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][1] * H;
                    double det = (x - x0)*(y1-y0) - (y - y0)*(x1-x0);
                    int sign = sgn(det);
                    if (prevSign == 0) prevSign = sign; else
                        if (sign == 0) sign = prevSign; else
                        if (sign != prevSign) {
                            isInside = false;
                            break;
                        }
                    prevSign = sign;
                    double edgeLen = sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0));
                    double distEdge = std::abs(det)/ edgeLen;
                    double dotp = (x - x0)*(x1 - x0) + (y - y0)*(y1 - y0);
                    if (dotp<0 || dotp>edgeLen*edgeLen) distEdge = 1E9;
                    mindistEdge = std::min(mindistEdge, distEdge);
                }
                if (isInside) {
                    if (i < N) {   // the N first particles may represent fluid, displayed in blue
                      image[((H - y - 1)*W + x) * 3] = 0;
                      image[((H - y - 1)*W + x) * 3 + 1] = 0;
                      image[((H - y - 1)*W + x) * 3 + 2] = 255;
                    }
                    if (mindistEdge <= 2) {
                        image[((H - y - 1)*W + x) * 3] = 0;
                        image[((H - y - 1)*W + x) * 3 + 1] = 0;
                        image[((H - y - 1)*W + x) * 3 + 2] = 0;
                    }
 
                }
                    
            }
        }
    }
    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, &image[0], 0);
}

class OptimalTransport {
public:
    OptimalTransport() {};
    void optimize() {
        vor.compute();
    }
    VoronoiDiagram vor;
};

class Fluid {
public:
    Fluid(int N = 1000):N(N){
        particles.resize(N);
        velocities.resize(N, Vector(0, 0, 0));

        double cx = 0.5, cy = 0.75; // Centered and higher up
        double radius = 0.15;

        for (int i = 0; i < N; i++) {
            double r = radius * sqrt(uniform(engine));
            double theta = uniform(engine) * 2 * M_PI;
            double x = cx + r * cos(theta);
            double y = cy + r * sin(theta);
            particles[i] = Vector(x, y, 0);
        }

        fluid_volume = VOL_FLUID;
        ot.vor.points = particles;
        ot.vor.weights.resize(N + 1, 1.0); // Add a boundary cell
        ot.vor.weights[N] = 0.99;
    }


    void time_step(double dt) {
        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81, 0);
        double m_i = 200;
        ot.vor.points = particles;
        ot.optimize();
        for (int i = 0; i < particles.size(); i++) {
            Vector center_cell = ot.vor.cells[i].centroid();
            Vector spring_force = (center_cell - particles[i]) / epsilon2;
            Vector all_forces = m_i * g + spring_force;
            velocities[i] = velocities[i] + dt / m_i * all_forces;
            particles[i] = particles[i] + (dt * velocities[i]);
            if (particles[i][0] < 0) {
                particles[i][0] = 0;
                velocities[i][0] *= -0.5;
            }
            if (particles[i][0] > 1) {
                particles[i][0] = 1;
                velocities[i][0] *= -0.5;
            }
            if (particles[i][1] < 0) {
                particles[i][1] = 0;
                velocities[i][1] *= -0.5;
            }
            if (particles[i][1] > 1) {
                particles[i][1] = 1;
                velocities[i][1] *= -0.5;
            }
        }

    }
    typedef Polygon Facet;

    void run_simulation() {
        double dt = 0.005;
        for (int i = 0; i < 100; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i, N);
        }
    }

    OptimalTransport ot;
    std::vector<Vector> particles;
    std::vector<Vector> velocities;
    int N;
    double fluid_volume;
};

// Main function
int main() {
    int N = 1000;
    Fluid fluid(N);
    fluid.run_simulation();
    return 0;
}
