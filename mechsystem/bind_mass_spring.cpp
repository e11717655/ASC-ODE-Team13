#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "mass_spring.hpp"
#include "Newmark.hpp"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<Mass<3>>);
PYBIND11_MAKE_OPAQUE(std::vector<Fix<3>>);
PYBIND11_MAKE_OPAQUE(std::vector<Spring>);

PYBIND11_MODULE(mass_spring, m) {
    m.doc() = "mass-spring-system simulator"; 

    // --- BASE CLASSES ---
    py::class_<NonlinearFunction, std::shared_ptr<NonlinearFunction>>(m, "NonlinearFunction");
    py::class_<Constraint>(m, "Constraint");

    // --- 2D CLASSES ---

    py::class_<Mass<2>> (m, "Mass2d")
      .def_property("mass",
                    [](Mass<2> & m) { return m.mass; },
                    [](Mass<2> & m, double mass) { m.mass = mass; })
      .def_property_readonly("pos",
                             [](Mass<2> & m) { return m.pos.data(); });
      ;
      
    m.def("Mass", [](double m, std::array<double,2> p)
    {
      return Mass<2>{m, { p[0], p[1] }};
    });

    py::class_<Fix<2>> (m, "Fix2d")
      .def_property_readonly("pos",
                             [](Fix<2> & f) { return f.pos.data(); });

    m.def("Fix", [](std::array<double,2> p)
    {
      return Fix<2>{ { p[0], p[1] } };
    });

    // DistanceConstraint Binding
    py::class_<DistanceConstraint<2>, Constraint>(m, "DistanceConstraint2d")
      .def(py::init<size_t, size_t, size_t, double>(), 
           py::arg("ndof"), py::arg("massA"), py::arg("massB"), py::arg("length"));


    // --- 3D CLASSES ---

    py::class_<Mass<3>> (m, "Mass3d")
      .def_property("mass",
                    [](Mass<3> & m) { return m.mass; },
                    [](Mass<3> & m, double mass) { m.mass = mass; })
      .def_property_readonly("pos",
                             [](Mass<3> & m) { return m.pos.data(); });
    ;

    m.def("Mass", [](double m, std::array<double,3> p)
    {
      return Mass<3>{m, { p[0], p[1], p[2] }};
    });


    py::class_<Fix<3>> (m, "Fix3d")
      .def_property_readonly("pos",
                             [](Fix<3> & f) { return f.pos.data(); });
    
    m.def("Fix", [](std::array<double,3> p)
    {
      return Fix<3>{ { p[0], p[1], p[2] } };
    });

    py::class_<Connector> (m, "Connector");

    py::class_<Spring> (m, "Spring")
      .def(py::init<double, double, std::array<Connector,2>>())
      .def_property_readonly("connectors",
                             [](Spring & s) { return s.connectors; })
      ;

    
    py::bind_vector<std::vector<Mass<3>>>(m, "Masses3d");
    py::bind_vector<std::vector<Fix<3>>>(m, "Fixes3d");
    py::bind_vector<std::vector<Spring>>(m, "Springs");        
    
    
    // MassSpringSystem2d (Updated with missing methods)
    py::class_<MassSpringSystem<2>> (m, "MassSpringSystem2d")
      .def(py::init<>())
      .def_property("gravity", [](MassSpringSystem<2> & mss) { return mss.getGravity(); },
                    [](MassSpringSystem<2> & mss, std::array<double,2> g) { mss.setGravity(Vec<2>{g[0],g[1]}); })
      .def("add", [](MassSpringSystem<2> & mss, Mass<2> m) { return mss.addMass(m); })
      .def("add", [](MassSpringSystem<2> & mss, Fix<2> f) { return mss.addFix(f); })
      .def("add", [](MassSpringSystem<2> & mss, Spring s) { return mss.addSpring(s); })
      ;

    // MSS_Function2d (Updated with corrected VectorView/MatrixView constructors)
    py::class_<MSS_Function<2>, std::shared_ptr<MSS_Function<2>>, NonlinearFunction>(m, "MSS_Function2d")
      .def(py::init<MassSpringSystem<2>&>())
      .def(py::init<MassSpringSystem<2>&, Constraint&>())
      
      .def("evaluate", [](MSS_Function<2> & self, py::array_t<double> x, py::array_t<double> f) {
          py::buffer_info bx = x.request();
          py::buffer_info bf = f.request();
          
          // FIX: Argument order is (size, pointer)
          VectorView<double> vx(bx.size, (double*)bx.ptr);
          VectorView<double> vf(bf.size, (double*)bf.ptr);
          
          self.evaluate(vx, vf);
      })
      
      .def("evaluateDeriv", [](MSS_Function<2> & self, py::array_t<double> x, py::array_t<double> df) {
          py::buffer_info bx = x.request();
          py::buffer_info bdf = df.request();
          size_t n = bx.size;
          
          // FIX: Argument order is (size, pointer) for Vector
          VectorView<double> vx(n, (double*)bx.ptr);
          
          // FIX: Argument order is (rows, cols, pointer) for Matrix
          MatrixView<double> mdf(n, n, (double*)bdf.ptr);
          
          self.evaluateDeriv(vx, mdf);
      });
      
        
    py::class_<MassSpringSystem<3>> (m, "MassSpringSystem3d")
      .def(py::init<>())
      .def("__str__", [](MassSpringSystem<3> & mss) {
        std::stringstream sstr;
        sstr << mss;
        return sstr.str();
      })
      .def_property("gravity", [](MassSpringSystem<3> & mss) { return mss.getGravity(); },
                    [](MassSpringSystem<3> & mss, std::array<double,3> g) { mss.setGravity(Vec<3>{g[0],g[1],g[2]}); })
      .def("add", [](MassSpringSystem<3> & mss, Mass<3> m) { return mss.addMass(m); })
      .def("add", [](MassSpringSystem<3> & mss, Fix<3> f) { return mss.addFix(f); })
      .def("add", [](MassSpringSystem<3> & mss, Spring s) { return mss.addSpring(s); })
      .def_property_readonly("masses", [](MassSpringSystem<3> & mss) -> auto& { return mss.masses(); })
      .def_property_readonly("fixes", [](MassSpringSystem<3> & mss) -> auto& { return mss.fixes(); })
      .def_property_readonly("springs", [](MassSpringSystem<3> & mss) -> auto& { return mss.springs(); })
      .def("__getitem__", [](MassSpringSystem<3> mss, Connector & c) {
        if (c.type==Connector::FIX) return py::cast(mss.fixes()[c.nr]);
        else return py::cast(mss.masses()[c.nr]);
      })
      
      .def("getState", [] (MassSpringSystem<3> & mss) {
        Vector<> x(3*mss.masses().size());
        Vector<> dx(3*mss.masses().size());
        Vector<> ddx(3*mss.masses().size());
        mss.getState (x, dx, ddx);
        return std::vector<double>(x);
      })

      .def("simulate", [](MassSpringSystem<3> & mss, double tend, size_t steps) {
        Vector<> x(3*mss.masses().size());
        Vector<> dx(3*mss.masses().size());
        Vector<> ddx(3*mss.masses().size());
        mss.getState (x, dx, ddx);

        auto mss_func = std::make_shared<MSS_Function<3>> (mss);
        auto mass = std::make_shared<IdentityFunction> (x.size());

        SolveODE_Alpha(tend, steps, 0.8, x, dx, ddx, mss_func, mass);

        mss.setState (x, dx, ddx);  
    });
}