#pragma once

struct Object
{
    //center Coordinates
    double x;
    double y;
    double z;
    Object() : x(0), y(0), z(0) {};
    Object(double x, double y) : x(x), y(y), z(0) {};
    Object(double x, double y, double z) : x(x), y(y), z(z) {};
};

struct Point : public Object
{
    Point(double x, double y) : Object(x,y) {};
    Point(double x, double y, double z) : Object(x,y,z) {};
    Point(Object obj) : Object(obj) {};
    Point() : Object() {};
};

struct Circle : public Object
{
    double radius;
    Circle(double x, double y, double radius) : Object(x,y), radius(radius) {};
};

struct Rectangle : public Object
{
    double dim_x;
    double dim_y;
    Point topLeft;
    Point topRight;
    Point botLeft;
    Point botRight;

    Rectangle(double x, double y, double dim_x, double dim_y) : Object(x,y), dim_x(dim_x), dim_y(dim_y) {
        topLeft = Point(x - dim_x/2, y + dim_y/2);
        topRight = Point(x + dim_x/2, y + dim_y/2);
        botLeft = Point(x - dim_x/2, y - dim_y/2);
        botRight = Point(x - dim_x/2, y + dim_y/2);
    };
};


