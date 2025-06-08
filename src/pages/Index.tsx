
import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Copy, Download, RefreshCw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

const Index = () => {
  const [latexCode, setLatexCode] = useState(`\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}

\\title{LaTeX Practice Document}
\\author{Your Name}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}
Welcome to LaTeX! This is a powerful typesetting system for mathematical documents.

\\section{Mathematics}
Here are some examples of mathematical expressions:

\\subsection{Inline Math}
The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

\\subsection{Display Math}
\\begin{equation}
\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}
\\end{equation}

\\subsection{Matrices}
\\begin{equation}
A = \\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}
\\end{equation}

\\section{Lists}
\\begin{itemize}
\\item First item
\\item Second item
\\item Third item
\\end{itemize}

\\end{document}`);

  const [renderedHtml, setRenderedHtml] = useState('');
  const { toast } = useToast();

  const templates = {
    'Basic Document': `\\documentclass{article}
\\usepackage{amsmath}

\\title{My Document}
\\author{Your Name}
\\date{\\today}

\\begin{document}
\\maketitle

\\section{Introduction}
Your content here...

\\end{document}`,
    
    'Math Heavy': `\\documentclass{article}
\\usepackage{amsmath, amsfonts, amssymb}

\\begin{document}

\\section{Complex Mathematics}

\\begin{align}
f(x) &= \\int_{-\\infty}^{x} e^{-t^2} dt \\\\
g(x) &= \\sum_{n=0}^{\\infty} \\frac{x^n}{n!} \\\\
h(x) &= \\lim_{n \\to \\infty} \\left(1 + \\frac{x}{n}\\right)^n
\\end{align}

\\begin{theorem}
For any real number $x$, we have $e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!}$.
\\end{theorem}

\\end{document}`,

    'Presentation': `\\documentclass{beamer}
\\usetheme{Madrid}

\\title{My Presentation}
\\author{Your Name}
\\date{\\today}

\\begin{document}

\\frame{\\titlepage}

\\begin{frame}
\\frametitle{Introduction}
\\begin{itemize}
\\item Point one
\\item Point two
\\item Point three
\\end{itemize}
\\end{frame}

\\begin{frame}
\\frametitle{Mathematics}
\\begin{equation}
E = mc^2
\\end{equation}
\\end{frame}

\\end{document}`
  };

  useEffect(() => {
    // Load MathJax
    const script = document.createElement('script');
    script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
    document.head.appendChild(script);

    const script2 = document.createElement('script');
    script2.id = 'MathJax-script';
    script2.async = true;
    script2.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    document.head.appendChild(script2);

    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }
    };

    return () => {
      document.head.removeChild(script);
      document.head.removeChild(script2);
    };
  }, []);

  const renderLatex = () => {
    // Convert basic LaTeX to HTML for preview
    let html = latexCode
      .replace(/\\documentclass\{[^}]+\}/g, '')
      .replace(/\\usepackage\{[^}]+\}/g, '')
      .replace(/\\begin\{document\}/g, '')
      .replace(/\\end\{document\}/g, '')
      .replace(/\\title\{([^}]+)\}/g, '<h1 class="text-3xl font-bold mb-4">$1</h1>')
      .replace(/\\author\{([^}]+)\}/g, '<p class="text-lg text-muted-foreground mb-2">By: $1</p>')
      .replace(/\\date\{([^}]+)\}/g, '<p class="text-sm text-muted-foreground mb-6">$1</p>')
      .replace(/\\maketitle/g, '')
      .replace(/\\section\{([^}]+)\}/g, '<h2 class="text-2xl font-semibold mt-6 mb-3">$1</h2>')
      .replace(/\\subsection\{([^}]+)\}/g, '<h3 class="text-xl font-medium mt-4 mb-2">$1</h3>')
      .replace(/\\begin\{itemize\}/g, '<ul class="list-disc list-inside mb-4">')
      .replace(/\\end\{itemize\}/g, '</ul>')
      .replace(/\\item\s+/g, '<li class="mb-1">')
      .replace(/\\begin\{equation\}/g, '<div class="my-4">$$')
      .replace(/\\end\{equation\}/g, '$$</div>')
      .replace(/\\begin\{align\}/g, '<div class="my-4">\\begin{align}')
      .replace(/\\end\{align\}/g, '\\end{align}</div>')
      .replace(/\\begin\{pmatrix\}/g, '\\begin{pmatrix}')
      .replace(/\\end\{pmatrix\}/g, '\\end{pmatrix}')
      .replace(/\\begin\{theorem\}/g, '<div class="border-l-4 border-primary pl-4 my-4 bg-muted/50 p-4 rounded"><strong>Theorem:</strong> ')
      .replace(/\\end\{theorem\}/g, '</div>')
      .replace(/\n\s*\n/g, '</p><p class="mb-4">')
      .replace(/^([^<])/gm, '<p class="mb-4">$1')
      .replace(/([^>])$/gm, '$1</p>');

    setRenderedHtml(html);

    // Re-render MathJax after updating content
    setTimeout(() => {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
      }
    }, 100);
  };

  useEffect(() => {
    renderLatex();
  }, [latexCode]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(latexCode);
    toast({
      title: "Copied!",
      description: "LaTeX code copied to clipboard",
    });
  };

  const downloadLatex = () => {
    const blob = new Blob([latexCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'document.tex';
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadTemplate = (templateName: string) => {
    setLatexCode(templates[templateName as keyof typeof templates]);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
              LaTeX Practice Studio
            </h1>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={copyToClipboard}>
                <Copy className="w-4 h-4 mr-2" />
                Copy
              </Button>
              <Button variant="outline" size="sm" onClick={downloadLatex}>
                <Download className="w-4 h-4 mr-2" />
                Download
              </Button>
              <Button variant="outline" size="sm" onClick={renderLatex}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        {/* Templates */}
        <Card className="mb-6 p-4">
          <h3 className="text-lg font-semibold mb-3">Quick Start Templates</h3>
          <div className="flex gap-2 flex-wrap">
            {Object.keys(templates).map((template) => (
              <Button
                key={template}
                variant="outline"
                size="sm"
                onClick={() => loadTemplate(template)}
              >
                {template}
              </Button>
            ))}
          </div>
        </Card>

        {/* Editor and Preview */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
          {/* Editor */}
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-3">LaTeX Editor</h3>
            <div className="h-full border rounded">
              <Editor
                height="100%"
                defaultLanguage="latex"
                value={latexCode}
                onChange={(value) => setLatexCode(value || '')}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  wordWrap: 'on',
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                }}
              />
            </div>
          </Card>

          {/* Preview */}
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-3">Live Preview</h3>
            <div className="h-full border rounded p-4 bg-white text-black overflow-auto">
              <div 
                dangerouslySetInnerHTML={{ __html: renderedHtml }}
                className="prose prose-sm max-w-none"
              />
            </div>
          </Card>
        </div>

        {/* Help Section */}
        <Card className="mt-6 p-4">
          <Tabs defaultValue="basics">
            <TabsList>
              <TabsTrigger value="basics">Basics</TabsTrigger>
              <TabsTrigger value="math">Math</TabsTrigger>
              <TabsTrigger value="formatting">Formatting</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basics" className="mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="font-semibold mb-2">Document Structure</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>\documentclass{'{article}'}</code> - Document type</li>
                    <li><code>\usepackage{'{amsmath}'}</code> - Load packages</li>
                    <li><code>\begin{'{document}'}</code> - Start content</li>
                    <li><code>\end{'{document}'}</code> - End content</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Sections</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>\section{'{Title}'}</code> - Main section</li>
                    <li><code>\subsection{'{Title}'}</code> - Subsection</li>
                    <li><code>\subsubsection{'{Title}'}</code> - Sub-subsection</li>
                  </ul>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="math" className="mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="font-semibold mb-2">Math Modes</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>$x^2$</code> - Inline math</li>
                    <li><code>$$x^2$$</code> - Display math</li>
                    <li><code>\begin{'{equation}'}</code> - Numbered equation</li>
                    <li><code>\begin{'{align}'}</code> - Multiple equations</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Common Symbols</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>\frac{'{a}{b}'}</code> - Fraction</li>
                    <li><code>\sqrt{'{x}'}</code> - Square root</li>
                    <li><code>\sum</code> - Summation</li>
                    <li><code>\int</code> - Integral</li>
                  </ul>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="formatting" className="mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="font-semibold mb-2">Text Formatting</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>\textbf{'{bold}'}</code> - Bold text</li>
                    <li><code>\textit{'{italic}'}</code> - Italic text</li>
                    <li><code>\underline{'{text}'}</code> - Underlined</li>
                    <li><code>\emph{'{emphasis}'}</code> - Emphasis</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Lists</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><code>\begin{'{itemize}'}</code> - Bullet list</li>
                    <li><code>\begin{'{enumerate}'}</code> - Numbered list</li>
                    <li><code>\item</code> - List item</li>
                  </ul>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
};

export default Index;
