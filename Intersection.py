# -------------------------------------------------------------------------------------------------------------------- #
#                               Praktikum "mathematische Modellierung am Recher 2"                                     #
#                                             Abschnitt 3: Raytracing                                                  #
#                                                                                                                      #
#                           Sean Genkel, André Wetzel, Marko Rubinić, Aron Bartmer-Freund                              #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np


class Point:
    def __init__(self, x=0, y=0, z=0):
        self.values = (x, y, z)
        self.x, self.y, self.z = x, y, z

    def __str__(self):
        return str(self.values)

    # __iter__ und __getitem__, damit Datenstruktur komfortabler verwendbar ist
    def __iter__(self):
        return self.values.__iter__()

    def __getitem__(self, item):
        return self.values[item]


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.values = (x, y, z)
        self.x, self.y, self.z = x, y, z

    def __iter__(self):
        return self.values.__iter__()

    def __getitem__(self, item):
        return self.values[item]

    def __str__(self):
        return str(self.values)

    # Betrag bzw. Länge
    def length(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    # Skalarprodukt
    def scalprod(self, b):
        return self.x * b.x + self.y * b.y + self.z * b.z

    # Winkel zwischen zwei Vektoren
    def angle_with(self, b):
        len_u = self.length()
        len_v = b.length()
        # Winkel in Radians und Grad
        angle_rad = np.arccos(self.scalprod(b)/(len_u * len_v))
        angle_deg = angle_rad * 180/np.pi
        return angle_rad, angle_deg

    # Differenz zwischen zwei Vektoren
    def delta(self, b):
        return Vector(b.x - self.x,
                      b.y - self.y,
                      b.z - self.z)


class Ray:
    # Gerade bzw. Strahl besteht aus Stützvektor (sup_vec) und Spannvektor (dir_vec) sowie dessen Faktor r
    # G = sup_vec + r * dir_vec
    def __init__(self, sup_vec, dir_vec, r=1):
        self.sup_vec = sup_vec
        self.dir_vec = dir_vec
        self.r = r

    # Für einen Faktor r einen konkreten Punkt auf der Gerade berechnen
    def calc_point(self, r):
        return Point(int(self.sup_vec.x + r * self.dir_vec.x),
                     int(self.sup_vec.y + r * self.dir_vec.y),
                     int(self.sup_vec.z + r * self.dir_vec.z))


class Plain:
    # Ebene besteht aus Stützvektor (sup_vec) und 2 Spannvektoren (dir_vec) sowie deren Faktoren r und s
    # E = sup_vec + r * dir_vec1 + s * dir_vec2
    def __init__(self, sup_vec, dir_vec1, dir_vec2, r=1, s=1):
        self.r = r
        self.s = s
        # Wenn Vektoren übergeben wurden, einfach als Vektoren für die Ebene übernehmen
        if isinstance(sup_vec, Vector):
            self.sup_vec = sup_vec
            self.dir_vec1 = dir_vec1
            self.dir_vec2 = dir_vec2
        # Wenn Punkte übergeben wurden, die Vektoren (0A, AB, AC) berechnen
        elif isinstance(sup_vec, Point):
            self.sup_vec = Vector(sup_vec.x, sup_vec.y, sup_vec.z)
            self.dir_vec1 = Vector(dir_vec1.x - sup_vec.x, dir_vec1.y - sup_vec.y, dir_vec1.z - sup_vec.z)
            self.dir_vec2 = Vector(dir_vec2.x - sup_vec.x, dir_vec2.y - sup_vec.y, dir_vec2.z - sup_vec.z)


class Sphere:
    def __init__(self, mid_point=(0, 0, 0), radius=1):
        self.mid_point = mid_point
        self.radius = radius
        # radius^2 in den Konstruktor packen, spart Rechenleistung
        self.radius_squared = radius**2

    def __str__(self):
        return "(x - " + str(self.mid_point[0]) + ")^2 + (y - " + str(self.mid_point[1]) + ")^2 + " \
               "(z - " + str(self.mid_point[2]) + ")^2 = " + str(self.radius**2)

# Schnittpunkt Gerade, Ebene
def intersect_ray_plain(ray, plain):

    # alle Vektoren mit Parameter auf eine Seite bringen, Gleichungssystem lösen

    a = np.array([[ray.dir_vec.x, -plain.dir_vec1.x, -plain.dir_vec2.x],
                  [ray.dir_vec.y, -plain.dir_vec1.y, -plain.dir_vec2.y],
                  [ray.dir_vec.z, -plain.dir_vec1.z, -plain.dir_vec2.z]])
    b = np.array([plain.sup_vec.x - ray.sup_vec.x,
                  plain.sup_vec.y - ray.sup_vec.y,
                  plain.sup_vec.z - ray.sup_vec.z])

    x = np.linalg.solve(a, b)

    # errechneten Parameter x[0] als Faktor in die Parameterform der Geraden einsetzen -> Schnittpunkt erhalten
    intersection = ray.calc_point(x[0])
    # Länge Vektor |AB| berechnen (A = Stützvektor Gerade, B = Schnittpunkt)
    dist_point_origin = Point(int(intersection.x - ray.sup_vec.x),
                              int(intersection.y - ray.sup_vec.y),
                              int(intersection.z - ray.sup_vec.z))
    # Ausgabe Schnittpunkt, Abstand
    return intersection, dist_point_origin

# Schnittpunkt Gerade, Dreieck
def intersect_ray_polygon(ray, plain):

    a = np.array([[ray.dir_vec.x, -plain.dir_vec1.x, -plain.dir_vec2.x],
                  [ray.dir_vec.y, -plain.dir_vec1.y, -plain.dir_vec2.y],
                  [ray.dir_vec.z, -plain.dir_vec1.z, -plain.dir_vec2.z]])
    b = np.array([plain.sup_vec.x - ray.sup_vec.x,
                  plain.sup_vec.y - ray.sup_vec.y,
                  plain.sup_vec.z - ray.sup_vec.z])

    x = np.linalg.solve(a, b)

    # Überprüfung ob der Schnittpunkt im Dreieck liegt. Parameter müssen <=1 sein.
    if x[0] <= 1 and x[1] <= 1 and x[0]+x[1] <= 1:
        intersection = ray.calc_point(x[0])
        # Länge Vektor |AB| berechnen (A = Stützvektor Gerade, B = Schnittpunkt)
        dist_point_origin = Point(intersection.x - ray.sup_vec.x,
                                  intersection.y - ray.sup_vec.y,
                                  intersection.z - ray.sup_vec.z)
        # Ausgabe Schnittpunkt, Abstand
        return intersection, dist_point_origin
    else:
        # liegt nicht im Dreieck
        return

# FUNKTIONIERT NICHT! TypeError: unsupported operand type(s) for *: 'Vector' and 'Vector'

    # Hilfreiche Quelle hierfür:
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

# Schnittpunkt Gerade, Kugel
def intersect_ray_sphere(ray, sphere):

    D = Vector(ray.dir_vec)
    L = Vector(ray.sup_vec)
    a = D.scalprod(D)
    b = 2 * D.scalprod(L)
    c = L.scalprod(L) - sphere.radius_squared

    d = (b**2) - (4*a*c)
    # Wenn Diskriminante d > 0 existieren 2 verschiedene Lösungen (Gerade durchstößt Kugel), für
    #                    d = 0 existieren 2 identische Lösungen (Gerade tangiert Kugel) und für
    if d >= 0:

        t1 = (-b - np.sqrt(d))/(2*a)
        t2 = (-b + np.sqrt(d))/(2*a)

        intersection1 = ray.calc_point(t1)
        intersection2 = ray.calc_point(t2)

    # Wenn Diskriminante d < 0 existieren keine Lösungen (Gerade schneidet Kugel nicht)
    else:
        intersection1 = None
        intersection2 = None

    return intersection1, intersection2


# -------------------------------------------------- Testeroni ------------------------------------------------------- #


g1 = Ray(Vector(2, -3, 2), Vector(1, -1, 3))
e1 = Plain(Vector(-3, 1, 1), Vector(1, -2, -1), Vector(0, -1, 2))


print("Schnittpunkt:", 6*"\t", intersect_ray_plain(g1, e1)[0],
      "\n" "Abstand:", 7*"\t", intersect_ray_plain(g1, e1)[1])


v1 = Vector(5, 4, 2)
v2 = Vector(-1, 3, 2)

print("v1: ", 8*"\t", v1.__str__())
print("v2: ", 8*"\t", v2.__str__())
print("Länge v1: ", 7*"\t", v1.length())
print("Skalarprodukt v1*v2:", 4*"\t", v1.scalprod(v2))
print("Winkel zw. v1/v2 (Radians, Grad):", "\t", v1.angle_with(v2))
print("Differenz zw. v1/v2:", 4*"\t", v1.delta(v2))

g2 = Ray(v1, v2)
mid_point = Point(0, 7, 7)
K1 = Sphere(mid_point, 5)

# Beispiel aus: https://www.lernhelfer.de/schuelerlexikon/mathematik-abitur/artikel/kugel-und-gerade)
#print(intersect_ray_sphere(g2, K1))  # müsste P1 = (2.5714.., 11.2857.., 6.8571..) und P2 = (4, 7, 4) zurückgeben
print("Kreisgleichung als String: ", 2*"\t", K1.__str__())

