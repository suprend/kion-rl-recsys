"""Synthetic, standard-library report tests; no TeX engine is executed."""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import report_tool as tool


def document(body: str, title: str = "Простой отчёт") -> str:
    return ("\\documentclass{article}\n\\usepackage{booktabs,graphicx,hyperref}\n"
            "\\begin{document}\n\\title{" + title + "}\n\\author{}\n\\date{}\n\\maketitle\n"
            + tool.BODY_BEGIN + "\n" + body + "\n" + tool.BODY_END + "\n\\end{document}\n")


class ConversionTests(unittest.TestCase):
    def test_relative_hyperlinks_are_root_artifact_relative_not_assets(self):
        for destination in ("REPRODUCING.md", "report-latex/evaluation64k.json",
                            "evidence/fullstudy/run_64k/metrics.json"):
            for command in ("href", "url"):
                body = "\\" + command + "{" + destination + "}"
                if command == "href":
                    body += "{evidence}"
                md, assets = tool.convert(document(body))
                self.assertIn("](" + destination + ")", md)
                self.assertEqual(assets, [])

    def test_relative_hyperlink_encoding_and_tex_punctuation(self):
        body = r"\href{evidence/run\_64k/result\%2Ejson}{evidence}"
        self.assertIn("[evidence](evidence/run_64k/result%2Ejson)",
                      tool.convert(document(body))[0])
        self.assertEqual(tool.hyperlink_destination("evidence/данные.json"),
                         "evidence/%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5.json")
        for destination in ("http://example.org/a", "https://example.org/a?b=1&c=2"):
            self.assertEqual(tool.hyperlink_destination(destination), destination)

    def test_relative_hyperlinks_reject_unsafe_or_ambiguous_paths(self):
        invalid = ("", "a" * 2049, "../secret", "a/../../secret", "a/../b.json", "./a.json",
                   "/tmp/a", "//server/share", r"a\b", "C:/a", "C:a", "a/b:stream",
                   "file:///secret", "javascript:alert(1)", "data:text/plain,test",
                   "a//b", "a/", "a?x=1", "a#fragment", "#fragment", "a. /b", "a./b",
                   "NUL.json", "a/COM1.txt", "a b", "a\x01b", "a\x7fb", "a\nb",
                   "%2e%2e/a", "a/%2E%2e/b", "%2fsecret", "a%2fb", "a%5cb",
                   "a%3Ab", "a%00b", "a%0Ab", "a%7fb", "a%C2%85b", "a%E2%80%8Bb", "%252e%252e/a",
                   "%2566ile%253a/a", "a%3fb", "a%23b", "a%20b", "a%", "a%zz", "a%ff",
                   r"\input{secret}", "https://user:password@example.org/x", "https://[broken")
        for destination in invalid:
            with self.subTest(destination=destination), self.assertRaises(tool.ReportError):
                tool.hyperlink_destination(destination)
        for destination in (r"\%2e\%2e/secret", r"a\%5cb", r"a\%00b", r"a\%252fb"):
            with self.subTest(tex=destination), self.assertRaises(tool.ReportError):
                tool.convert(document(r"\href{" + destination + "}{bad}"))

    def test_ragged_columns_and_row_gap(self):
        template = (r"\begin{longtable}{SPEC}\toprule H \\ \midrule X \\GAP"
                    r"\bottomrule\end{longtable}")
        spec = r">{\raggedright\arraybackslash}p{4cm}"
        body = template.replace("SPEC", spec).replace("GAP", "[4pt]")
        self.assertIn("| H |\n| :--- |\n| X |", tool.convert(document(body))[0])
        for bad in (r">{\input{x}}p{4cm}", r">{\raggedright\arraybackslash}l",
                    r">{\raggedright\arraybackslash}"):
            with self.subTest(spec=bad), self.assertRaises(tool.ReportError):
                tool.convert(document(template.replace("SPEC", bad).replace("GAP", "")))
        for gap in ("[-4pt]", "[40pt]", r"[\input{x}]", "[4pt+1pt]"):
            with self.subTest(gap=gap), self.assertRaises(tool.ReportError):
                tool.convert(document(template.replace("SPEC", spec).replace("GAP", gap)))

    def test_longtable_identical_repeated_header_and_no_break_rows(self):
        header = r"\toprule Name & Value \\* \midrule"
        body = (r"\begin{longtable}{lr}" + header + r"\endfirsthead " + header
                + r"\endhead A & 1 \\*[4pt] B & 2 \\ \bottomrule\end{longtable}")
        simple = (r"\begin{longtable}{lr}" + header
                  + r" A & 1 \\ B & 2 \\ \bottomrule\end{longtable}")
        self.assertEqual(tool.convert(document(body)), tool.convert(document(simple)))
        for invalid in (body.replace("longtable", "tabular"),
                        body.replace(r"\endhead", ""),
                        body.replace(r"\endfirsthead", ""),
                        body.replace(r"\endhead", r"\endhead\endhead"),
                        body.replace(r"\endfirsthead", r"\endfirsthead\endfirsthead"),
                        body.replace("Name & Value", "Different & Value", 1),
                        body.replace(r"\endhead", r"\endfoot"),
                        body.replace(r"\\*[4pt]", r"\\**[4pt]"),
                        body.replace(r"\\*[4pt]", r"\\*[-4pt]")):
            with self.subTest(body=invalid), self.assertRaises(tool.ReportError):
                tool.convert(document(invalid))

    def test_fontspec_cmu_configuration_is_preamble_metadata(self):
        original = document("Содержимое.")
        font = (r"\usepackage{fontspec}" + "\n"
                + r"\setmainfont[BoldFont=cmunbx.otf,ItalicFont=cmunti.otf,BoldItalicFont=cmunbi.otf]{cmunrm.otf}" + "\n")
        source = original.replace(r"\begin{document}", font + r"\begin{document}")
        self.assertEqual(tool.convert(source), tool.convert(original))

    def test_fixed_preamble_page_break_penalties(self):
        original = document("Text.")
        source = original.replace(r"\begin{document}",
                                  "\\widowpenalty=10000\n\\clubpenalty=10000\n\\begin{document}")
        self.assertEqual(tool.convert(source), tool.convert(original))
        for setting in ("=10001", "=100000", "=-10000", "=10000junk", r"=\input{x}", "{10000}"):
            with self.subTest(setting=setting), self.assertRaises(tool.ReportError):
                tool.convert(original.replace(r"\begin{document}",
                             "\\widowpenalty" + setting + "\n\\begin{document}"))
        with self.assertRaises(tool.ReportError):
            tool.convert(document(r"\clubpenalty=10000 Text."))

    def test_display_math_keeps_formula_and_rejects_commands(self):
        body = "Before.\\[\n" + r"\max_\theta\;\operatorname{softmax}(x\mid y)" + "\n" + r"\Vert\pi\]After."
        md, _ = tool.convert(document(body))
        self.assertIn("Before.\n\n$$\n" + r"\max_\theta\;\operatorname{softmax}(x\mid y) \Vert\pi" + "\n$$\n\nAfter.", md)
        for body in (r"\[\input{x}\]", r"\[x", r"\[\]", r"\[{x\]", r"\[x$y\]",
                     r"\[x% comment" + "\n" + r"y\]", r"\textbf{\[x\]}",
                     r"\[\operatorname{\input{x}}\]", r"\[\newcommand{x}{y}\]"):
            with self.subTest(body=body), self.assertRaises(tool.ReportError):
                tool.convert(document(body))

    def test_fontspec_rejects_paths_macros_unknown_or_missing_styles(self):
        valid = r"\setmainfont[BoldFont=cmunbx.otf,ItalicFont=cmunti.otf,BoldItalicFont=cmunbi.otf]{cmunrm.otf}"
        invalid = [valid.replace("{cmunrm.otf}", "{../cmunrm.otf}"),
                   valid.replace("{cmunrm.otf}", "{Arial}"),
                   valid.replace("BoldFont=cmunbx.otf", "BoldFont=../cmunbx.otf"),
                   valid.replace("BoldFont=cmunbx.otf", r"BoldFont=\input{x}"),
                   valid.replace("BoldFont=cmunbx.otf", "Path=fonts/,BoldFont=cmunbx.otf"),
                   valid.replace("BoldFont=cmunbx.otf", "BoldFont=cmunbx.otf,BoldFont=cmunbx.otf"),
                   valid.replace("ItalicFont=cmunti.otf,", ""),
                   valid.replace("[BoldFont=cmunbx.otf,ItalicFont=cmunti.otf,BoldItalicFont=cmunbi.otf]", "")]
        for command in invalid:
            source = document("Text.").replace(r"\begin{document}", r"\usepackage{fontspec}" + "\n" + command + "\n" + r"\begin{document}")
            with self.subTest(command=command), self.assertRaises(tool.ReportError):
                tool.convert(source)
        for prefix in [valid + "\n", r"\usepackage{fontspec}" + "\n"]:
            with self.subTest(prefix=prefix), self.assertRaises(tool.ReportError):
                tool.convert(document("Text.").replace(r"\begin{document}", prefix + r"\begin{document}"))

    def test_prose_links_math_and_nested_formats(self):
        md, assets = tool.convert(document(r"""\emph{Подзаголовок.}

\section{Содержание}
Текст \textbf{жирный с \emph{акцентом} и \(\beta=0{,}5\)}.
\href{https://example.org/a?b=1&c=2}{ссылка}; \texttt{watched\_pct/100}; 82\%.

\subsection{Математика}
$1/\mathrm{rank}$ и \([-0{,}1;\,+0{,}2]\)."""))
        self.assertTrue(md.startswith(tool.NOTICE + "\n\n# Простой отчёт\n"))
        self.assertIn("## Содержание", md)
        self.assertIn("**жирный с *акцентом* и $\\beta=0{,}5$**", md)
        self.assertIn("[ссылка](https://example.org/a?b=1&c=2)", md)
        self.assertIn("`watched_pct/100`", md)
        self.assertIn("82%", md)
        self.assertEqual(assets, [])

    def test_list_table_and_figure(self):
        md, assets = tool.convert(document(r"""\begin{enumerate}
\item Один.
\item Два \textbf{слова}.
\end{enumerate}

\begin{center}\begin{tabular}{lr}
\toprule
Модель & MAP \\
\midrule
\textbf{C} & 0,0969 \\
RL & 0,0944 \\
\bottomrule
\end{tabular}\end{center}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{images/test.pdf}
\caption{Результат \emph{проверки}.}
\end{figure}"""))
        self.assertIn("1. Один.\n2. Два **слова**.", md)
        self.assertIn("| Модель | MAP |\n| :--- | ---: |\n| **C** | 0,0969 |", md)
        self.assertIn("[Рисунок: Результат *проверки*.](report-latex/images/test.pdf)", md)
        self.assertEqual(assets, [Path("images/test.pdf")])

    def test_balanced_groups_comments_and_code_fence(self):
        md, _ = tool.convert(document("A% removed\nB \\textbf{C {D}} \\texttt{a`b}."))
        self.assertIn("AB **C D** ``a`b``.", md)

    def test_newpage_is_ignored_at_block_boundaries(self):
        md, _ = tool.convert(document(r"Before.\newpage\section{Next}" + "\nAfter."))
        self.assertIn("Before.\n\n## Next\n\nAfter.", md)
        self.assertEqual(tool.convert(document(r"\newpage Text.\newpage")),
                         tool.convert(document("Text.")))
        for body in (r"\textbf{A\newpage B}", r"\newpage{argument}",
                     r"\newpage[option]", "\\newpage\n{argument}",
                     r"\begin{tabular}{l}\toprule H \\ \midrule X\newpage \\ \bottomrule\end{tabular}"):
            with self.subTest(body=body), self.assertRaises(tool.ReportError):
                tool.convert(document(body))

    def test_fixed_width_tables_preserve_content_and_alignment(self):
        for env in ("tabular", "longtable"):
            for width in ("3cm", "25.5mm", ".5pt", "2 cm"):
                body = (r"\begin{" + env + "}{l p{" + width + r"}cr}\toprule "
                        + r"Feature & Meaning & Type & Count \\ \midrule "
                        + "\\texttt{user\\_id} & Длинное описание\nна двух строках "
                        + r"с \textbf{акцентом} и $x$ & ID & 3 \\ \bottomrule\end{" + env + "}")
                source = document(body).replace("booktabs,graphicx,hyperref", "booktabs,graphicx,hyperref,longtable")
                with self.subTest(env=env, width=width):
                    md, assets = tool.convert(source)
                    self.assertIn("| :--- | :--- | :---: | ---: |", md)
                    self.assertIn("| `user_id` | Длинное описание на двух строках с **акцентом** и $x$ | ID | 3 |", md)
                    self.assertEqual(assets, [])

    def test_fixed_width_tables_reject_unsupported_specs_and_rows(self):
        for env in ("tabular", "longtable"):
            template = (r"\begin{" + env + r"}{SPEC}\toprule H \\ \midrule X \\ \bottomrule\end{" + env + "}")
            for spec in ("", "p", "p{}", "p{0cm}", "p{-2cm}", "p{1e2pt}", "p{1in}",
                         r"p{\textwidth}", r"p{2\input{x}}", "p{2cm+1pt}", "p{NaNcm}",
                         "p{2cm}junk", "|l|", "*{2}{l}", "m{2cm}", r">{\raggedright}p{2cm}"):
                with self.subTest(env=env, spec=spec), self.assertRaises(tool.ReportError):
                    tool.convert(document(template.replace("SPEC", spec)))
            valid = template.replace("SPEC", "p{2cm}")
            for invalid in (valid.replace("X", r"\multicolumn{1}{l}{X}"),
                            valid.replace(r"\midrule", r"\midrule\endhead"),
                            valid.replace("X", "X & Y"),
                            valid.replace("X", r"\texttt{x|y}"),
                            valid.replace(r"\bottomrule", ""),
                            valid.replace(r"\end{" + env + "}", r"\end{other}")):
                with self.subTest(env=env, invalid=invalid), self.assertRaises(tool.ReportError):
                    tool.convert(document(invalid))

    def test_preserve_spaces_inside_plain_groups(self):
        md, _ = tool.convert(document(r"A{ B }C and \$price\$."))
        self.assertIn(r"A B C and \$price\$.", md)

    def test_nested_lists_and_literal_markdown_punctuation(self):
        md, _ = tool.convert(document(r"""\begin{itemize}
\item Parent.
\begin{enumerate}\item Child.\end{enumerate}
\item Next.
\end{itemize}

\# Header \_literal\_ and \texttt{`quoted`}."""))
        self.assertIn("- Parent.\n  \n  1. Child.\n- Next.", md)
        self.assertIn(r"\# Header \_literal\_", md)
        self.assertIn("`` `quoted` ``", md)

    def test_reject_unknown_or_malformed_constructs(self):
        invalid = [r"\input{secret}", r"\foo{lost}", r"\textbf{open", r"\(\input{x}\)",
                   r"\begin{unknown}x\end{unknown}", r"\texttt{\emph{x}}", r"\texttt{\href{https://example.org}{x}}", r"a & b", r"\end{itemize}",
                   r"\begin{itemize}\item x", r"\begin{itemize}\item x\end{itemize}\item outside",
                   r"\href{file:///secret}{bad}", r"\href{javascript:alert(1)}{bad}",
                   r"\begin{tabular}{ll}\toprule A & B \\ \midrule X \\ \bottomrule\end{tabular}",
                   r"\begin{figure}\includegraphics{../secret.pdf}\caption{x}\end{figure}"]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(tool.ReportError):
                tool.convert(document(body))

    def test_reject_metadata_macros_and_duplicate_markers(self):
        for source in [document("x", r"\input{x}"), document("x") + tool.BODY_END,
                       document("x").replace("\\usepackage{booktabs,graphicx,hyperref}", "\\input{x}"),
                       document("x").replace("booktabs,graphicx,hyperref", "unknownpackage"),
                       document("x").replace("\\end{document}", "\\end{document}x")]:
            with self.subTest(source=source), self.assertRaises(tool.ReportError):
                tool.convert(source)

    def test_reject_tex_character_encoding_before_parsing(self):
        encoded = r"^^5cinput"
        sources = [document(r"\(" + encoded + "{secret}\\)"),
                   document("Text.").replace("\\documentclass{article}", "\\documentclass[" + encoded + "]{article}"),
                   document("Text.").replace("\\usepackage{booktabs,graphicx,hyperref}", "\\usepackage[" + encoded + "]{booktabs}"),
                   document("Text.").replace("\\begin{document}", "% " + encoded + "\n\\begin{document}")]
        for source in sources:
            with self.subTest(source=source), self.assertRaisesRegex(tool.ReportError, r"\^\^ character encodings"):
                tool.convert(source)

    def test_table_rejects_math_pipe_without_altering_formula_elsewhere(self):
        def table(cell):
            return r"\begin{tabular}{l}\toprule H \\ \midrule " + cell + r" \\ \bottomrule\end{tabular}"
        for cell in [r"$|x|$", r"\(|x|\)", r"\textbf{\(\left|x\right|\)}", r"\texttt{a|b}"]:
            with self.subTest(cell=cell), self.assertRaisesRegex(tool.ReportError, "Unescaped pipes in table cells"):
                tool.convert(document(table(cell)))
        md, _ = tool.convert(document(r"$|x|$ and \(\left|x\right|\)."))
        self.assertIn(r"$|x|$ and $\left|x\right|$.", md)
        md, _ = tool.convert(document(table("a|b")))
        self.assertIn(r"| a\|b |", md)


class FilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "repo"
        self.root.mkdir()
        (self.root / "report-latex").mkdir()
        self.source = self.root / tool.SOURCE
        self.source.write_text(document("Текст."), encoding="utf-8")
        self.compiler = self.base / "tectonic.exe"
        self.compiler.write_bytes(b"synthetic executable, never run")
        self.cache = self.base / "cache"
        self.cache.mkdir()
        self.work = self.base / "work"
        self.output = self.base / "result.pdf"

    def build(self, **kwargs):
        return tool.build(self.root, self.compiler, self.output, self.work, cache_dir=self.cache, **kwargs)

    def fake_compile(self, command, **kwargs):
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["timeout"], 120)
        self.assertIn("--untrusted", command)
        self.assertIn("--only-cached", command)
        out = Path(command[command.index("--outdir") + 1])
        (out / "main.pdf").write_bytes(b"%PDF-1.7\nsynthetic\n%%EOF\n")
        self.assertEqual(Path(kwargs["env"]["TECTONIC_CACHE_DIR"]), self.cache)
        staged = Path(command[-1])
        self.assertEqual(staged.read_bytes(), self.source.read_bytes())
        if os.name != "nt":
            self.assertFalse(staged.stat().st_mode & stat.S_IWUSR)
        return subprocess.CompletedProcess(command, 0)

    def test_markdown_check_never_writes(self):
        before = self.source.read_bytes()
        with self.assertRaises(tool.ReportError):
            tool.markdown(self.root, True)
        self.assertFalse((self.root / tool.MARKDOWN).exists())
        tool.markdown(self.root, False)
        target = self.root / tool.MARKDOWN
        info = target.stat()
        tool.markdown(self.root, True)
        self.assertEqual(target.stat().st_mtime_ns, info.st_mtime_ns)
        self.assertEqual(self.source.read_bytes(), before)
        target.write_text("stale", encoding="utf-8")
        with self.assertRaises(tool.ReportError):
            tool.markdown(self.root, True)
        self.assertEqual(target.read_text(encoding="utf-8"), "stale")

    def test_build_stages_read_only_and_creates_new_output(self):
        with patch.object(tool.subprocess, "run", side_effect=self.fake_compile) as run:
            receipt = self.build()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(receipt["status"], "compiled-unreviewed")
        self.assertTrue((self.work / "build-receipt.json").is_file())
        self.assertEqual(self.output.read_bytes(), b"%PDF-1.7\nsynthetic\n%%EOF\n")

    def test_relative_hyperlinks_do_not_stage_or_read_linked_files(self):
        self.source.write_text(document(r"\href{evidence/fullstudy/metrics.json}{evidence}"),
                               encoding="utf-8")
        with patch.object(tool.subprocess, "run", side_effect=self.fake_compile):
            receipt = self.build()
        self.assertEqual(set(receipt["inputs"]), {"report-latex/main.tex"})
        self.assertEqual([p.name for p in (self.work / "source").iterdir()], ["main.tex"])

    def test_refuse_existing_output_or_work_without_running_compiler(self):
        self.output.write_bytes(b"keep")
        with patch.object(tool.subprocess, "run") as run, self.assertRaises(tool.ReportError):
            self.build()
        self.assertFalse(run.called)
        self.assertFalse(self.work.exists())
        self.assertEqual(self.output.read_bytes(), b"keep")
        self.output.unlink()
        self.work.mkdir()
        with patch.object(tool.subprocess, "run") as run, self.assertRaises(tool.ReportError):
            self.build()
        self.assertFalse(run.called)

    def test_refuse_in_repo_and_parent_traversal_paths(self):
        for output, work in [(self.root / "new.pdf", self.work), (self.output, self.root / "work"),
                             (self.base / "missing" / ".." / "new.pdf", self.work)]:
            with self.subTest(output=output, work=work), patch.object(tool.subprocess, "run") as run, self.assertRaises(tool.ReportError):
                tool.build(self.root, self.compiler, output, work, cache_dir=self.cache)
            self.assertFalse(run.called)

    def test_missing_graphics_refuses_before_work_creation(self):
        self.source.write_text(document(r"\begin{figure}\includegraphics{images/missing.png}\caption{x}\end{figure}"), encoding="utf-8")
        with patch.object(tool.subprocess, "run") as run, self.assertRaises(tool.ReportError):
            self.build()
        self.assertFalse(run.called)
        self.assertFalse(self.work.exists())

    def test_encoded_commands_refused_before_staging_or_execution(self):
        self.source.write_text(document(r"\(^^5cinput{secret}\)"), encoding="utf-8")
        with patch.object(tool.subprocess, "run") as run, self.assertRaisesRegex(tool.ReportError, r"\^\^ character encodings"):
            self.build()
        self.assertFalse(run.called)
        self.assertFalse(self.work.exists())
        self.assertFalse(self.output.exists())

    def test_compiler_failure_preserves_logs_and_has_no_pdf(self):
        with patch.object(tool.subprocess, "run", return_value=subprocess.CompletedProcess(["fake"], 7)), self.assertRaises(tool.ReportError):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertIn('"returncode": 7', (self.work / "build-receipt.json").read_text(encoding="utf-8"))

    def test_timeout_records_failure_without_output(self):
        with patch.object(tool.subprocess, "run", side_effect=subprocess.TimeoutExpired(["fake"], 120)), self.assertRaises(tool.ReportError):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertIn("TimeoutExpired", (self.work / "build-receipt.json").read_text(encoding="utf-8"))

    def test_output_collision_after_compilation_is_not_overwritten(self):
        def collision(command, **kwargs):
            result = self.fake_compile(command, **kwargs)
            self.output.write_bytes(b"concurrent owner")
            return result
        with patch.object(tool.subprocess, "run", side_effect=collision), self.assertRaises(tool.ReportError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"concurrent owner")

    def test_explicit_download_opt_in_only_changes_child_environment(self):
        original = os.environ.get("TECTONIC_CACHE_DIR")
        def compile_with_resources(command, **kwargs):
            self.assertNotIn("--only-cached", command)
            (self.work / "output" / "main.pdf").write_bytes(b"%PDF-1.7\n")
            return subprocess.CompletedProcess(command, 0)
        with patch.object(tool.subprocess, "run", side_effect=compile_with_resources):
            receipt = self.build(allow_resource_downloads=True)
        self.assertTrue(receipt["allow_resource_downloads"])
        self.assertEqual(os.environ.get("TECTONIC_CACHE_DIR"), original)

    def test_reject_source_symlink_where_supported(self):
        original = self.source.read_bytes()
        self.source.unlink()
        elsewhere = self.base / "elsewhere.tex"
        elsewhere.write_bytes(original)
        try:
            self.source.symlink_to(elsewhere)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable")
            raise
        with self.assertRaises(tool.ReportError):
            tool.inspect_source(self.root)


if __name__ == "__main__":
    unittest.main()
