from streamlit.components.v1 import html

def render_3dmol(pdb_text, pred_range, iedb_ranges=None):
    ps, pe = pred_range

    def js_range(start, end):
        return f"Array.from({{length:{end-start+1}}},(_,i)=>i+{start})"

    iedb_js = ""
    if iedb_ranges:
        for s, e in iedb_ranges:
            iedb_js += f"""
            viewer.addStyle({{resi:{js_range(s,e)}}},
                            {{cartoon:{{color:'blue'}},stick:{{radius:0.2}}}});
            """

    html_code = f"""
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width:100%;height:600px"></div>
    <script>
      let viewer = $3Dmol.createViewer("viewer",{{backgroundColor:'#0E1117'}});
      viewer.addModel(`{pdb_text}`,"pdb");
      viewer.setStyle({{}},{{cartoon:{{color:'#AAAAAA',opacity:0.4}}}});
      viewer.addStyle({{resi:{js_range(ps,pe)}}},
                      {{cartoon:{{color:'red'}},stick:{{radius:0.3}}}});
      {iedb_js}
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    html(html_code, height=600)
