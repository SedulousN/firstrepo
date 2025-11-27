# app.py
# Streamlit app to display & share 6 standalone C++ algorithm programs
# Run: pip install streamlit
#      streamlit run app.py

import streamlit as st
import io
import zipfile
from textwrap import dedent

st.set_page_config(page_title="C++ Algorithms Collection", layout="wide")

st.title("C++ Algorithms Collection")
st.markdown(
    """
This app contains six self-contained C++ programs.  
Click the file name to expand and view the code. Use the **copy** icon in the code viewer or the **Download** buttons to get the source files.  
You can also download all six files as a single ZIP.
"""
)

# -- C++ program sources ----------------------------------------------------
max_flow_code = dedent(r'''
// max_flow_edmonds_karp.cpp
#include <bits/stdc++.h>
using namespace std;

struct MaxFlow {
    struct Edge { int to; int cap; int rev; };
    int n;
    vector<vector<Edge>> g;
    MaxFlow(int n): n(n), g(n) {}
    void addEdge(int u, int v, int c){
        g[u].push_back({v,c,(int)g[v].size()});
        g[v].push_back({u,0,(int)g[u].size()-1});
    }
    int maxflow(int s, int t){
        int flow = 0;
        while(true){
            vector<int> parent(n, -1), pe(n, -1);
            queue<int> q; q.push(s); parent[s]=s;
            while(!q.empty() && parent[t]==-1){
                int u=q.front(); q.pop();
                for(int i=0;i<(int)g[u].size();++i){
                    auto &e = g[u][i];
                    if(parent[e.to]==-1 && e.cap>0){
                        parent[e.to]=u; pe[e.to]=i; q.push(e.to);
                    }
                }
            }
            if(parent[t]==-1) break;
            int aug = INT_MAX;
            int v = t;
            while(v!=s){ int u = parent[v]; aug = min(aug, g[u][pe[v]].cap); v=u; }
            v = t;
            while(v!=s){ int u = parent[v]; auto &e = g[u][pe[v]]; e.cap -= aug; g[v][e.rev].cap += aug; v=u; }
            flow += aug;
        }
        return flow;
    }
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // Example graph:
    // 0 -> 1 (3), 0 -> 2 (2), 1 -> 2 (1), 1 -> 3 (2), 2 -> 3 (4)
    MaxFlow mf(4);
    mf.addEdge(0,1,3);
    mf.addEdge(0,2,2);
    mf.addEdge(1,2,1);
    mf.addEdge(1,3,2);
    mf.addEdge(2,3,4);
    cout << "Max flow from 0 to 3 = " << mf.maxflow(0,3) << "\n"; // expected 4
    return 0;
}
''').strip()

bipartite_matching_code = dedent(r'''
// bipartite_matching_kuhn.cpp
#include <bits/stdc++.h>
using namespace std;

struct Kuhn {
    int nL, nR;
    vector<vector<int>> adj;
    vector<int> matchR, matchL;
    Kuhn(int nL, int nR): nL(nL), nR(nR), adj(nL), matchR(nR, -1), matchL(nL, -1) {}
    void addEdge(int uLeft, int vRight){ adj[uLeft].push_back(vRight); }
    bool dfs(int v, vector<char>& seen){
        if(seen[v]) return false;
        seen[v]=1;
        for(int to: adj[v]){
            if(matchR[to] == -1 || dfs(matchR[to], seen)){
                matchR[to] = v;
                matchL[v] = to;
                return true;
            }
        }
        return false;
    }
    int maxMatching(){
        int matches = 0;
        for(;;){
            bool progress = false;
            vector<char> seen(nL,0);
            for(int v=0; v<nL; ++v){
                if(matchL[v]==-1 && dfs(v, seen)){ progress = true; ++matches; }
            }
            if(!progress) break;
        }
        return matches;
    }
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // Example: Left nodes 0..2, Right 0..1
    Kuhn km(3,2);
    km.addEdge(0,0);
    km.addEdge(0,1);
    km.addEdge(1,0);
    km.addEdge(2,1);
    int m = km.maxMatching();
    cout << "Max matching size = " << m << "\n";
    for(int i=0;i<3;i++){
        if(km.matchL[i] != -1) cout << "Left " << i << " matched to Right " << km.matchL[i] << "\n";
    }
    return 0;
}
''').strip()

min_vertex_cover_code = dedent(r'''
// min_vertex_cover_bipartite.cpp
#include <bits/stdc++.h>
using namespace std;

// We'll reuse Kuhn implementation for matching
struct Kuhn {
    int nL, nR;
    vector<vector<int>> adj;
    vector<int> matchR, matchL;
    Kuhn(int nL, int nR): nL(nL), nR(nR), adj(nL), matchR(nR, -1), matchL(nL, -1) {}
    void addEdge(int uLeft, int vRight){ adj[uLeft].push_back(vRight); }
    bool dfs(int v, vector<char>& seen){
        if(seen[v]) return false;
        seen[v]=1;
        for(int to: adj[v]){
            if(matchR[to] == -1 || dfs(matchR[to], seen)){
                matchR[to] = v;
                matchL[v] = to;
                return true;
            }
        }
        return false;
    }
    int maxMatching(){
        int matches=0;
        for(;;){
            bool prog=false; vector<char> seen(nL,0);
            for(int v=0; v<nL; ++v){
                if(matchL[v]==-1 && dfs(v,seen)){ prog=true; ++matches; }
            }
            if(!prog) break;
        }
        return matches;
    }
};

pair<vector<int>, vector<int>> minVertexCover(Kuhn &km){
    int nL = km.nL, nR = km.nR;
    vector<char> visL(nL,false), visR(nR,false);
    queue<int> q;
    // start from unmatched left vertices
    for(int i=0;i<nL;++i) if(km.matchL[i] == -1){ q.push(i); visL[i]=true; }
    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v: km.adj[u]){
            if(!visR[v] && km.matchL[u] != v){
                visR[v] = true;
                if(km.matchR[v] != -1 && !visL[km.matchR[v]]){
                    visL[km.matchR[v]] = true;
                    q.push(km.matchR[v]);
                }
            }
        }
    }
    // min vertex cover = (L \\ Z) U (R ∩ Z) where Z = visited vertices in alternating graph
    vector<int> coverL, coverR;
    for(int i=0;i<nL;++i) if(!visL[i]) coverL.push_back(i);
    for(int j=0;j<nR;++j) if(visR[j]) coverR.push_back(j);
    return {coverL, coverR};
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Kuhn km(3,2);
    km.addEdge(0,0);
    km.addEdge(0,1);
    km.addEdge(1,0);
    km.addEdge(2,1);
    cout << "Max matching size = " << km.maxMatching() << "\n";
    auto cover = minVertexCover(km);
    cout << "Min Vertex Cover Left indices: ";
    for(int x: cover.first) cout << x << " ";
    cout << "\nMin Vertex Cover Right indices: ";
    for(int x: cover.second) cout << x << " ";
    cout << "\n";
    return 0;
}
''').strip()

tsp_code = dedent(r'''
// tsp_nearest_neighbor.cpp
#include <bits/stdc++.h>
using namespace std;

double dist(pair<double,double> a, pair<double,double> b){
    double dx=a.first-b.first, dy=a.second-b.second;
    return sqrt(dx*dx + dy*dy);
}

vector<int> nearest_neighbor(const vector<pair<double,double>>& pts, int start=0){
    int n = pts.size();
    if(n==0) return {};
    vector<char> used(n,false);
    vector<int> tour; tour.reserve(n);
    int cur = start; used[cur]=true; tour.push_back(cur);
    for(int k=1;k<n;++k){
        int best=-1;
        double bestd = 1e300;
        for(int i=0;i<n;++i) if(!used[i]){
            double d = dist(pts[cur], pts[i]);
            if(d < bestd){ bestd = d; best = i; }
        }
        cur = best; used[cur]=true; tour.push_back(cur);
    }
    return tour;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<pair<double,double>> pts = {{0,0},{1,2},{2,1},{5,0},{3,3}};
    auto tour = nearest_neighbor(pts, 0);
    cout << "Nearest neighbor tour order: ";
    for(int v: tour) cout << v << " ";
    cout << "\nTour length (closed): ";
    double length = 0;
    for(size_t i=0;i<tour.size();++i){
        int a=tour[i], b=tour[(i+1)%tour.size()];
        length += dist(pts[a], pts[b]);
    }
    cout << length << "\n";
    return 0;
}
''').strip()

segment_intersection_code = dedent(r'''
// segment_intersections.cpp
#include <bits/stdc++.h>
using namespace std;

struct P { double x,y; };
P operator-(const P&a,const P&b){ return {a.x-b.x, a.y-b.y}; }
double cross(const P&a,const P&b){ return a.x*b.y - a.y*b.x; }

bool onSegment(const P &a, const P &b, const P &p){
    double minx=min(a.x,b.x), maxx=max(a.x,b.x);
    double miny=min(a.y,b.y), maxy=max(a.y,b.y);
    return fabs(cross(b-a, p-a)) < 1e-9 && p.x >= minx-1e-9 && p.x <= maxx+1e-9 && p.y >= miny-1e-9 && p.y <= maxy+1e-9;
}

bool segInter(const P &a1, const P &a2, const P &b1, const P &b2, P &out){
    P r = a2 - a1;
    P s = b2 - b1;
    double rxs = cross(r,s);
    double qpxr = cross(b1 - a1, r);
    if(fabs(rxs) < 1e-12){
        if(fabs(qpxr) < 1e-12){
            // collinear: check overlap (return one point of overlap if exists)
            vector<P> pts = {a1,a2,b1,b2};
            sort(pts.begin(), pts.end(), [](const P&a,const P&b){
                if(fabs(a.x-b.x) > 1e-9) return a.x < b.x;
                return a.y < b.y;
            });
            for(auto &p: pts) if(onSegment(a1,a2,p) && onSegment(b1,b2,p)){ out = p; return true; }
            return false;
        }
        return false;
    } else {
        double t = cross((b1 - a1), s) / rxs;
        double u = cross((b1 - a1), r) / rxs;
        if(t >= -1e-9 && t <= 1+1e-9 && u >= -1e-9 && u <= 1+1e-9){
            out = { a1.x + t * r.x, a1.y + t * r.y };
            return true;
        }
        return false;
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<pair<P,P>> segs = {
        {{0,0},{3,3}},
        {{0,3},{3,0}},
        {{4,4},{5,5}},
        {{1,1},{2,2}}
    };
    vector<tuple<int,int,P>> inters;
    for(int i=0;i<(int)segs.size();++i){
        for(int j=i+1;j<(int)segs.size();++j){
            P out;
            if(segInter(segs[i].first, segs[i].second, segs[j].first, segs[j].second, out))
                inters.emplace_back(i,j,out);
        }
    }
    cout << "Intersections found: " << inters.size() << "\n";
    for(auto &t: inters){
        int i,j; P p; tie(i,j,p) = t;
        cout << "Segments " << i << " & " << j << " intersect at (" << p.x << "," << p.y << ")\n";
    }
    return 0;
}
''').strip()

convex_hull_code = dedent(r'''
// convex_hull_graham.cpp
#include <bits/stdc++.h>
using namespace std;

double orient(const pair<double,double>& a, const pair<double,double>& b, const pair<double,double>& c){
    return (b.first-a.first)*(c.second-a.second) - (b.second-a.second)*(c.first-a.first);
}
double dist2(const pair<double,double>& a, const pair<double,double>& b){
    double dx=a.first-b.first, dy=a.second-b.second; return dx*dx+dy*dy;
}

vector<pair<double,double>> grahamScan(vector<pair<double,double>> pts){
    int n = pts.size();
    if(n <= 1) return pts;
    // pivot = lowest y (then lowest x)
    int pid = min_element(pts.begin(), pts.end(), [](auto &p1, auto &p2){
        if(p1.second != p2.second) return p1.second < p2.second;
        return p1.first < p2.first;
    }) - pts.begin();
    swap(pts[0], pts[pid]);
    auto pivot = pts[0];
    sort(pts.begin()+1, pts.end(), [&](const auto &A, const auto &B){
        double o = orient(pivot, A, B);
        if(fabs(o) < 1e-12) return dist2(pivot, A) < dist2(pivot, B);
        return o > 0;
    });
    vector<pair<double,double>> st;
    for(auto &p: pts){
        while(st.size() >= 2 && orient(st[st.size()-2], st.back(), p) <= 0) st.pop_back();
        st.push_back(p);
    }
    return st;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<pair<double,double>> pts = {{0,0},{1,1},{2,2},{0,2},{2,0},{1,0.5},{3,1}};
    auto hull = grahamScan(pts);
    cout << "Convex hull points (CCW):\n";
    for(auto &p: hull) cout << "(" << p.first << "," << p.second << ") ";
    cout << "\n";
    return 0;
}
''').strip()

files = {
    "max_flow_edmonds_karp.cpp": max_flow_code,
    "bipartite_matching_kuhn.cpp": bipartite_matching_code,
    "min_vertex_cover_bipartite.cpp": min_vertex_cover_code,
    "tsp_nearest_neighbor.cpp": tsp_code,
    "segment_intersections.cpp": segment_intersection_code,
    "convex_hull_graham.cpp": convex_hull_code,
}

# -- UI layout --------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Files")
    selected = st.selectbox("Choose a file to view", list(files.keys()))
    st.markdown("**Quick actions**")
    if st.button("Download all as ZIP"):
        # prepare zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, src in files.items():
                zf.writestr(fname, src)
        buf.seek(0)
        st.download_button("Download ZIP", buf, file_name="cpp_algorithms_collection.zip", mime="application/zip")

    st.markdown("---")
    st.write("Files included:")
    for fname in files:
        st.write(f"- `{fname}`")

with col2:
    st.header("Code viewer & download")
    code = files[selected]
    st.subheader(selected)
    # syntax-highlighted code block (Streamlit provides copy icon)
    st.code(code, language='cpp')

    # download single file
    st.download_button(
        label="Download this file",
        data=code.encode('utf-8'),
        file_name=selected,
        mime="text/x-c++src"
    )

st.markdown("---")
st.header("How to use")
st.markdown(dedent("""
- Each file is a self-contained C++ program with a small `main()` example.
"""))

st.markdown("Made with ❤️ — edit and share freely.")
