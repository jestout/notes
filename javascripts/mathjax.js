window.MathJax = {
    loader: {load: ['[tex]/boldsymbol', '[tex]/ams']},
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      packages: {'[+]': ['boldsymbol', 'ams']},
      macros: {
        mb: ['{\\mathbf{#1}}', 1],
        ud: ['{\\mathrm{d}}'],
        bm: ['{\\boldsymbol{#1}}', 1],
        lab: ['{\\mathrm{#1}}', 1]
      }
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    },
  };
  

  document$.subscribe(() => { 
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })