#include <iostream>
#include <cstdlib>
// A simple class with a constuctor and some methods...
class Foo
{
    public:
        Foo(int);
        void bar();
        int foobar(int, int);
        void barfoo();
    private:
        int val;
        int * arr;
};
Foo::Foo(int n)
{
    val = n;
    // arr = new int[n];
    arr = (int *) calloc(n, sizeof (int));


}
void Foo::bar()
{
    std::cout << "Length is " << val << std::endl;
    for (int i = 0; i< val; i++){
        std::cout << arr[i] << "\t";
    }
    std::cout << "\n";
}
int Foo::foobar(int n, int i)
{
    arr[i] = n;
    return val + n;
}
void Foo::barfoo()
{
    free(arr);
    
}


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    Foo* Foo_new(int n) {return new Foo(n);}
    void Foo_bar(Foo* foo) {foo->bar();}
    int Foo_foobar(Foo* foo, int n, int i) {return foo->foobar(n, i);}
}
