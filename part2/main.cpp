#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "codes/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "codes/stb_image.h"

#include "vector.cpp"

static std::default_random_engine engine(10); 
static std::uniform_real_distribution<double> uniform(0, 1);

// if the Polygon class name conflicts with a class in wingdi.h on Windows, use a namespace or change the name
class Polygon {  
public:
    std::vector<Vector> vertices;
};  

class VoronoiDiagram {
    public:

    VoronoiDiagram() {};

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
                V = clip_by_bisector(V, points[i], points[j]);
            }
            cells[i] = V;
        }
    }
    std::vector<Vector> points;
    std::vector<Polygon> cells;
};

 
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
 
//static int progress


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
    int N = 100;
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