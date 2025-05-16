import ttl;

int main() {
    ttl::index<"i"> i;
    ttl::index<"j"> j;

    int a[16]{};
    ttl::tspan A(a, 4, 4);
    auto B = A(i,j);
}
