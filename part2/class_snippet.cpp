#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "codes/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "codes/stb_image.h"

#include "codes/vector.cpp"

#include <sstream>
#include<random>
#include "codes/lbfgs.h"


static std::default_random_engine engine(10); 
static std::uniform_real_distribution<double> uniform(0, 1);

#define M_PI 3.14159265358979323846264338327950288
#define VOL_FLUID 0.6

// if the Polygon class name conflicts with a class in wingdi.h on Windows, use a namespace or change the name
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
            c = c + (vertices[i] + vertices[ip]*crossP);
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

    std::vector<Vector> vertices;
};  

//something

class VoronoiDiagram {
    public:
        int N_disk = 100;

    VoronoiDiagram() {
        unit_disk.resize(N_disk);
        for(int i=0; i<N_disk; i++) {
            double theta = i/(double)N_disk;
            unit_disk[i] = Vector(sin(-theta), cos(-theta), 0);
        }
    };

    std::vector<Vector> unit_disk;

    Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {
        Vector N(v[1]-u[1], u[0]-v[0], 0);

        Polygon result;
        result.vertices;
        for(int i=0; i< V.vertices.size(); i++) {

            const Vector& A = V.vertices[(i==0)? V.vertices.size() - 1: i-1];
            const Vector& B = V.vertices[i];

            if (dot(u-B, N) >=0) { //B inside

                if(dot(u-A, N) <0) { // A outside
                    Vector M = (P0 + Pi) * 0.5;
                    double t = dot(u-A, N)/dot(B-A, N);
                    Vector P = A + t * (B-A);
    
                    result.vertices.push_back(P);
                }
                result.vertices.push_back(B);
            }else {
                if(dot(u-A, N) >=0) {
                    Vector M = (P0 + Pi) * 0.5;
                    double t = dot(M-A, Pi-P0)/dot((B-A), (Pi-P0));
                    Vector P = A + t * (B-A);

                    result.vertices.push_back(P);
                }
            }
        }
        return result;  // Add this return statement
    }

    Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi) {
        Polygon result;
        const Vector P0Pi = Pi - P0;
        for(int i=0; i< V.vertices.size(); i++) {

            const Vector& A = V.vertices[(i==0)? V.vertices.size() - 1: i-1];
            const Vector& B = V.vertices[i];

            if ((B-P0).norm2() <= (B-Pi).norm2()) { //B inside

                if((A-P0).norm2() > (A-Pi).norm2()) { // A outside
                    Vector M = (P0 + Pi) * 0.5;
                    double t = dot(M-A, P0Pi)/dot(B-A, P0Pi);
                    Vector P = A + t * (B-A);
    
                    result.vertices.push_back(P);
                }
                result.vertices.push_back(B);
            }else {
                if((A-P0).norm2() <= (A-Pi).norm2()) {
                    Vector M = (P0 + Pi) * 0.5;
                    double t = dot(M-A, Pi-P0)/dot((B-A), (Pi-P0));
                    Vector P = A + t * (B-A);

                    result.vertices.push_back(P);
                }
            }
        }
        return result;  // Add this return statement
    }

    void compute() {
        Polygon square;
        square.vertices.push_back(Vector(0, 0));
        square.vertices.push_back(Vector(0, 1));
        square.vertices.push_back(Vector(1, 1));
        square.vertices.push_back(Vector(1, 0));

        cells.resize(points.size());

//#pragma omp parallel for schedule(dynamic, 1)
        for(int i=0; i<points.size(); i++) {
            Polygon V = square;
            for (int j=0; j<points.size(); j++) {
                if(i==j) continue;
                V = clip_by_bisector(V, points[i], points[j], weights[i], weights[j]);
            }

            //clip V by disk 
            for (int j=0; j<N_disk; j++) {
                double radius = sqrt(weights[i] - weights[weights.size() - 1]);
                Vector u = unit_disk[j]*radius + points[i];
                Vector v = unit_disk[(j+1)%N_disk]*radius + points[i];
                V = clip_by_edge(V, u, v);
            }

            cells[i] = V;
        }
    }
    std::vector<Vector> points;
    std::vector<Polygon> cells;
    std::vector<double> weights;
};

//void Optim

int sgn(double x) {
    if (x > 0) return 1;
    if (x<0) return -1;
    return 0;
}

 
// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");  
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0]*1000., 1000.-(*points)[i][1]*1000);
        }
        fprintf(f, "</g>\n");
    }

    fprintf(f, "</svg>\n");
    fclose(f);
}
 
void save_frame(const std::vector<Facet> &cells, std::string filename, int frameid = 0) {
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
//static int progress

class OptimalTransport {
    public:
        OptimalTransport() {};

        void optimize();

        VoronoiDiagram vor;
};

class Fluid {
    public:
    Fluid(int N = 1000):N(N){
        particles.resize(N);
        velocities.resize(N, Vector(0, 0, 0));
        for (int i = 0; i<0; i++) {
            particles[i] = Vector(uniform(engine), uniform(engine), uniform(engine));
        }
        fluid_volume = VOL_FLUID;

        ot.vor.points = particles;
        ot.vor.weights.resize(N+1);
        std::fill(ot.vor.weights.begin(), ot.vor.weights.end(), 1);
        ot.vor.weights[N] = 0.99;
    };

    void time_step(double dt) {

        double epsilon2 = 0.004*0.004;
        Vector g(0, -9.81, 0);
        double m_i = 200;

        ot.vor.points = particles;
        ot.optimize();

        for (int i=0; i<particles.size(); i++) {

            Vector center_cell = ot.vor.cells[i].centroid();
            Vector spring_force = (center_cell - particles[i])/epsilon2;
            Vector all_forces = m_i*g + spring_force;
            velocities[i] = velocities[i] +  dt / m_i * all_forces;
            particles[i] = particles[i] + (dt * velocities);

        }
    }

    void run_simulation() {
        double dt = 0;

        for (int i=0; i<100; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i);
        }
    }

    void init() {

    }

    OptimalTransport ot;
    std::vector<Vector> particles;
    std::vector<Vector> velocities;
    int N;
    double fluid_volume;
    std::vector<Polygon> cells;
    std::vector<double> weights;
};


// Adds one frame of an animated svg file. frameid is the frame number (between 0 and nbframes-1).
// polygons is a list of polygons, describing the current frame.
// The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg_animated(const std::vector<Polygon> &polygons, std::string filename, int frameid, int nbframes) {
    FILE* f;
    if (frameid == 0) {
        f = fopen(filename.c_str(), "w+");
        fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
        fprintf(f, "<g>\n");
    } else {
        f = fopen(filename.c_str(), "a+");
    }
    fprintf(f, "<g>\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<polygon points = \""); 
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000-polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"none\" stroke = \"black\"/>\n");
    }
    fprintf(f, "<animate\n");
    fprintf(f, "    id = \"frame%u\"\n", frameid);
    fprintf(f, "    attributeName = \"display\"\n");
    fprintf(f, "    values = \"");
    for (int j = 0; j < nbframes; j++) {
        if (frameid == j) {
            fprintf(f, "inline");
        } else {
            fprintf(f, "none");
        }
        fprintf(f, ";");
    }
    fprintf(f, "none\"\n    keyTimes = \"");
    for (int j = 0; j < nbframes; j++) {
        fprintf(f, "%2.3f", j / (double)(nbframes));
        fprintf(f, ";");
    }
    fprintf(f, "1\"\n   dur = \"5s\"\n");
    fprintf(f, "    begin = \"0s\"\n");
    fprintf(f, "    repeatCount = \"indefinite\"/>\n");
    fprintf(f, "</g>\n");
    if (frameid == nbframes - 1) {
        fprintf(f, "</g>\n");
        fprintf(f, "</svg>\n");
    }
    fclose(f);
}

// Change void main() to int main()
int main() {

    Fluid.fluid(100);
    Fluid.run_simulation();
    exit(1);

    int N = 1000;
    VoronoiDiagram Vor;

    // Assuming uniform and engine are defined in vector.cpp
    for(int i = 0; i<N; i++) {
        Vor.points.push_back(Vector(uniform(engine), uniform(engine), 0));
    }
    Vor.compute();

    save_svg(Vor.cells, "testOut.svg", &Vor.points);  // Added points to visualize them
    return 0;
}

//g++ -std=c++17 main.cpp -o main
//./main