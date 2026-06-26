from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.libqsturng import qsturng
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="Solver Estatística Experimental API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def determinar_sig_texto(p_valor, gl_num, gl_den):
    if pd.isna(p_valor) or gl_num <= 0 or gl_den <= 0 or p_valor > 1: return "-" if gl_num <= 0 else "ns"
    if p_valor < 0.01: return "1% (**)"
    elif p_valor < 0.05: return "5% (*)"
    else: return "ns"

def classificar_cv(cv):
    if cv < 10: return "Ótimo"
    elif cv < 20: return "Bom"
    elif cv < 30: return "Regular"
    else: return "Ruim"

def calcular_f_tab(gl_num, gl_den, alpha=0.95):
    if gl_num > 0 and gl_den > 0 and not pd.isna(gl_den): 
        return round(stats.f.ppf(alpha, gl_num, gl_den), 2)
    return "-"

def formatar_f_calc(valor):
    """Formata F calculado: 2 casas decimais; científica somente em valores extremos."""
    if valor in ["-", None]:
        return "-"
    try:
        v = float(valor)
    except Exception:
        return valor
    if np.isnan(v) or np.isinf(v):
        return "-"
    if v == 0:
        return "0,00"
    if abs(v) >= 1e6 or abs(v) <= 1e-5:
        return f"{v:.2e}".replace(".", ",")
    return f"{v:.2f}".replace(".", ",")

def aplicar_letras(medias_series, dms_val=None, duncan_dict=None):
    trats = list(medias_series.index)
    vals = list(medias_series.values)
    n = len(vals)
    grupos = []
    
    for i in range(n):
        grupo_atual = [i]
        for j in range(i+1, n):
            diff = vals[i] - vals[j]
            dms = dms_val if dms_val is not None else duncan_dict[j - i + 1]
            if diff <= dms: grupo_atual.append(j)
            
        is_subset = False
        for g in grupos:
            if set(grupo_atual).issubset(set(g)):
                is_subset = True; break
        if not is_subset: grupos.append(grupo_atual)
            
    letras = ["" for _ in range(n)]
    letra_char = ord('a')
    for g in grupos:
        l = chr(letra_char)
        for idx in g: letras[idx] += l
        letra_char += 1
        
    return [{"Nível": str(trats[i]), "Média": round(vals[i], 2), "Letra": letras[i]} for i in range(n)]


# ================= MÓDULO 1: SIMPLES (DIC/DBC/DQL) =================
@app.post("/api/analise/simples")
async def analisar_simples(file: UploadFile = File(...), tipo_delineamento: str = Form("dbc"), tipo_teste: str = Form("anova"), modelo_regr: str = Form("linear"), testemunha: str = Form("")):
    try:
        content = await file.read()
        try: df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
        
        cols = list(df.columns)
        if tipo_delineamento == "dql": df.rename(columns={cols[0]: 'Trat', cols[1]: 'Linha', cols[2]: 'Coluna', cols[3]: 'Valor'}, inplace=True)
        else: df.rename(columns={cols[0]: 'Trat', cols[1]: 'Bloco', cols[2]: 'Valor'}, inplace=True)
            
        df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
        mg = df['Valor'].mean()
        sq_tot = ((df['Valor'] - mg)**2).sum()
        n_total = len(df)
        nT = len(df['Trat'].unique())
        rep_media = n_total / nT
        gl_tot = n_total - 1
        
        if tipo_delineamento == "dql":
            sq_trat = nT * ((df.groupby('Trat')['Valor'].mean() - mg)**2).sum()
            sq_linha = nT * ((df.groupby('Linha')['Valor'].mean() - mg)**2).sum()
            sq_col = nT * ((df.groupby('Coluna')['Valor'].mean() - mg)**2).sum()
            sq_res = max(0, sq_tot - sq_trat - sq_linha - sq_col)
            gl_trat, gl_linha, gl_col = nT - 1, nT - 1, nT - 1
            gl_res = (nT - 1) * (nT - 2)
            qm_trat, qm_linha, qm_col = sq_trat / gl_trat, sq_linha / gl_linha, sq_col / gl_col
            qm_res = sq_res / gl_res if gl_res > 0 else 0
            
            f_trat = qm_trat/qm_res if qm_res>0 else 0
            f_linha = qm_linha/qm_res if qm_res>0 else 0
            f_col = qm_col/qm_res if qm_res>0 else 0
            p_trat = 1 - stats.f.cdf(f_trat, gl_trat, gl_res)
            p_linha = 1 - stats.f.cdf(f_linha, gl_linha, gl_res)
            p_col = 1 - stats.f.cdf(f_col, gl_col, gl_res)
            
            anova = [
                {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": formatar_f_calc(f_trat), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(p_trat, gl_trat, gl_res)},
                {"FV": "Linhas", "GL": gl_linha, "SQ": round(sq_linha,2), "QM": round(qm_linha,2), "F Calc": formatar_f_calc(f_linha), "F Tab": calcular_f_tab(gl_linha, gl_res), "Sig": determinar_sig_texto(p_linha, gl_linha, gl_res)},
                {"FV": "Colunas", "GL": gl_col, "SQ": round(sq_col,2), "QM": round(qm_col,2), "F Calc": formatar_f_calc(f_col), "F Tab": calcular_f_tab(gl_col, gl_res), "Sig": determinar_sig_texto(p_col, gl_col, gl_res)},
                {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
                {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]
        else:
            nR = len(df['Bloco'].unique()) if tipo_delineamento == "dbc" else 1
            sq_trat = rep_media * ((df.groupby('Trat')['Valor'].mean() - mg)**2).sum()
            gl_trat = nT - 1
            
            if tipo_delineamento == "dbc":
                sq_bloc = nT * ((df.groupby('Bloco')['Valor'].mean() - mg)**2).sum()
                sq_res = max(0, sq_tot - sq_trat - sq_bloc)
                gl_bloc = nR - 1
                gl_res = gl_tot - gl_trat - gl_bloc
                qm_trat, qm_res = sq_trat/gl_trat, sq_res/gl_res if gl_res>0 else 0
                f_trat, f_bloc = qm_trat/qm_res if qm_res>0 else 0, (sq_bloc/gl_bloc)/qm_res if qm_res>0 else 0
                p_trat = 1 - stats.f.cdf(f_trat, gl_trat, gl_res)
                p_bloc = 1 - stats.f.cdf(f_bloc, gl_bloc, gl_res)
                anova = [
                    {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": formatar_f_calc(f_trat), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(p_trat, gl_trat, gl_res)},
                    {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(sq_bloc/gl_bloc,2), "F Calc": formatar_f_calc(f_bloc), "F Tab": calcular_f_tab(gl_bloc, gl_res), "Sig": determinar_sig_texto(p_bloc, gl_bloc, gl_res)},
                    {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
                    {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
                ]
            else: 
                sq_res = max(0, sq_tot - sq_trat)
                gl_res = gl_tot - gl_trat
                qm_trat, qm_res = sq_trat/gl_trat, sq_res/gl_res if gl_res>0 else 0
                f_trat = qm_trat/qm_res if qm_res>0 else 0
                p_trat = 1 - stats.f.cdf(f_trat, gl_trat, gl_res)
                anova = [
                    {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": formatar_f_calc(f_trat), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(p_trat, gl_trat, gl_res)},
                    {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
                    {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
                ]
        
        cv_val = round((np.sqrt(qm_res)/mg)*100, 2) if qm_res > 0 else 0
        cv_string = f"{cv_val}% ({classificar_cv(cv_val)})"

        if tipo_teste == "anova":
            return {"status": "sucesso", "cv": cv_string, "anova": anova}

        # === POST-HOC MÓDULO 1 ===
        alpha = 0.01 if p_trat < 0.01 else 0.05
        alpha_txt = "1%" if p_trat < 0.01 else "5%"
        ep = np.sqrt(qm_res / rep_media)
        medias = df.groupby('Trat')['Valor'].mean().sort_values(ascending=False)

        if "regr" in tipo_teste:
            if pd.to_numeric(df['Trat'], errors='coerce').isna().any(): 
                return {"status": "erro", "mensagem": "A regressão exige doses numéricas quantitativas."}
            x, y = np.array(medias.index, dtype=float), np.array(medias.values, dtype=float)
            y_mean = np.mean(y)
            fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(x, y, color='#0f172a', edgecolor='white', linewidth=1.2, label='Médias', zorder=5)
            x_plot = np.linspace(min(x), max(x), 100)
            
            x_otimo, y_otimo = None, None
            if modelo_regr == "linear":
                s, i, r, _, _ = stats.linregress(x, y); y_pred = i + s * x
                ax.plot(x_plot, i + s * x_plot, color='#2d7244', linewidth=3, label='Linear', zorder=4)
                eq, r2, gl_reg = f"y = {i:.4f} {'+' if s>=0 else '-'} {abs(s):.4f}x", r**2, 1
            elif modelo_regr == "quadratica":
                c = np.polyfit(x, y, 2); p = np.poly1d(c); y_pred = p(x)
                ax.plot(x_plot, p(x_plot), color='#2d7244', linewidth=3, label='Quadrática', zorder=4)
                eq, r2, gl_reg = f"y = {c[2]:.4f} {'+' if c[1]>=0 else '-'} {abs(c[1]):.4f}x {'+' if c[0]>=0 else '-'} {abs(c[0]):.4f}x²", 1 - (np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)), 2
                if c[0] != 0:
                    x_otimo = -c[1] / (2 * c[0]); y_otimo = p(x_otimo)
                    ax.axvline(x=x_otimo, color='#94a3b8', linestyle=(0, (4, 4)), alpha=0.85)
                    ax.scatter(x_otimo, y_otimo, color='#b18356', marker='*', s=220, edgecolor='white', linewidth=1.0, zorder=6, label=f'Dose Ótima: {x_otimo:.2f}')

            sq_reg_raw = rep_media * np.sum((y_pred - y_mean)**2); sq_desvio_raw = max(0, sq_trat - sq_reg_raw)
            gl_desvio = gl_trat - gl_reg
            qm_reg = sq_reg_raw / gl_reg if gl_reg > 0 else 0
            qm_desvio = sq_desvio_raw / gl_desvio if gl_desvio > 0 else 0
            f_reg = qm_reg / qm_res if qm_res > 0 else 0
            f_desvio = qm_desvio / qm_res if qm_res > 0 else 0
            p_reg = 1 - stats.f.cdf(f_reg, gl_reg, gl_res) if gl_reg > 0 else np.nan
            p_desvio = 1 - stats.f.cdf(f_desvio, gl_desvio, gl_res) if gl_desvio > 0 else np.nan
            
            anova_reg = [
                {"FV": f"Regressão ({modelo_regr.capitalize()})", "GL": gl_reg, "SQ": round(sq_reg_raw, 2), "QM": round(qm_reg, 2) if gl_reg>0 else "-", "F Calc": formatar_f_calc(f_reg) if gl_reg>0 else "-", "F Tab": calcular_f_tab(gl_reg, gl_res), "Sig": determinar_sig_texto(p_reg, gl_reg, gl_res)},
                {"FV": "Desvios da Regressão", "GL": gl_desvio, "SQ": round(sq_desvio_raw, 2), "QM": round(qm_desvio, 2) if gl_desvio>0 else "-", "F Calc": formatar_f_calc(f_desvio) if gl_desvio>0 else "-", "F Tab": calcular_f_tab(gl_desvio, gl_res), "Sig": determinar_sig_texto(p_desvio, gl_desvio, gl_res)},
                {"FV": "Total dos Tratamentos", "GL": gl_trat, "SQ": round(sq_trat, 2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]
            ax.set_title('Ajuste de regressão', fontsize=15, fontweight='bold', color='#0f172a'); ax.grid(True, color='#e2e8f0', linewidth=1); ax.set_facecolor('#fbfdfb'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
            b_png, b_pdf, b_svg = io.BytesIO(), io.BytesIO(), io.BytesIO()
            fig.savefig(b_png, format="png", bbox_inches='tight')
            fig.savefig(b_pdf, format="pdf", bbox_inches='tight')
            fig.savefig(b_svg, format="svg", bbox_inches='tight')
            plt.close(fig)
            
            res_dict = {"status": "sucesso", "tipo": "regressao", "equacao": eq, "r2": f"{r2:.4f}", "modelo": modelo_regr.capitalize(), "img_png": base64.b64encode(b_png.getvalue()).decode('utf-8'), "img_pdf": base64.b64encode(b_pdf.getvalue()).decode('utf-8'), "img_svg": base64.b64encode(b_svg.getvalue()).decode('utf-8'), "anova_reg": anova_reg}
            if x_otimo is not None: res_dict.update({"dose_otima": f"{x_otimo:.2f}", "resposta_otima": f"{y_otimo:.2f}"})
            return res_dict

        if "tukey" in tipo_teste:
            q_crit = qsturng(1 - alpha, nT, gl_res); dms = q_crit * ep
            return {"status": "sucesso", "tipo": "tukey", "nome_teste": "Tukey", "alpha_txt": alpha_txt, "q": round(q_crit, 4), "dms": round(dms, 4), "tabela": aplicar_letras(medias, dms_val=dms)}
        
        if "duncan" in tipo_teste:
            rp_duncan = {p: round(qsturng((1 - alpha)**(p-1), p, gl_res) * ep, 4) for p in range(2, nT + 1)}
            return {"status": "sucesso", "tipo": "duncan", "nome_teste": "Duncan", "alpha_txt": alpha_txt, "alcances": rp_duncan, "tabela": aplicar_letras(medias, duncan_dict=rp_duncan)}

        if "dunnett" in tipo_teste:
            if not testemunha or testemunha not in medias.index: return {"status": "erro", "mensagem": f"Testemunha '{testemunha}' não encontrada."}
            media_test = medias[testemunha]
            dms_dunnett = stats.t.ppf(1 - (alpha / (2 * (nT - 1))), gl_res) * ep * np.sqrt(2)
            res = [{"Tratamento": str(n), "Média": round(v, 2), "Diferença": round(v - media_test, 2), "Sig": "Significativo (*)" if abs(v - media_test) >= dms_dunnett else "ns"} for n, v in medias.items() if n != testemunha]
            return {"status": "sucesso", "tipo": "dunnett", "alpha_txt": alpha_txt, "dms": round(dms_dunnett, 4), "testemunha": testemunha, "media_testemunha": round(media_test, 2), "resultados": res}

    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ================= MÓDULO 2: FATORIAL DUPLO =================
@app.post("/api/analise/fatorial")
async def analisar_fatorial(file: UploadFile = File(...), tipo_teste: str = Form("anova"), modelo_regr: str = Form("linear"), testemunha: str = Form("")):
    try:
        content = await file.read()
        try: df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
        cols = list(df.columns)
        df.rename(columns={cols[0]: 'A', cols[1]: 'B', cols[2]: 'Bloco', cols[3]: 'Valor'}, inplace=True)
        df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
        
        nA, nB, nR = len(df['A'].unique()), len(df['B'].unique()), len(df['Bloco'].unique())
        mg = df['Valor'].mean()
        sq_tot = ((df['Valor'] - mg)**2).sum()
        sq_bloc = (nA*nB) * ((df.groupby('Bloco')['Valor'].mean() - mg)**2).sum()
        sq_a = (nB*nR) * ((df.groupby('A')['Valor'].mean() - mg)**2).sum()
        sq_b = (nA*nR) * ((df.groupby('B')['Valor'].mean() - mg)**2).sum()
        sq_inter = nR * ((df.groupby(['A', 'B'])['Valor'].mean() - mg)**2).sum() - sq_a - sq_b
        
        sq_trat = sq_a + sq_b + sq_inter
        gl_trat = (nA * nB) - 1
        qm_trat = sq_trat / gl_trat
        
        sq_res = max(0, sq_tot - sq_bloc - sq_a - sq_b - sq_inter)
        gl_a, gl_b, gl_inter, gl_bloc, gl_tot = nA-1, nB-1, (nA-1)*(nB-1), nR-1, (nA*nB*nR)-1
        gl_res = gl_tot - gl_a - gl_b - gl_inter - gl_bloc
        qm_res = sq_res/gl_res if gl_res > 0 else 0
        
        f_trat = qm_trat / qm_res if qm_res > 0 else 0
        p_trat = 1 - stats.f.cdf(f_trat, gl_trat, gl_res)
        p_inter = 1 - stats.f.cdf((sq_inter/gl_inter)/qm_res if qm_res>0 else 0, gl_inter, gl_res)

        anova = [
            {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": formatar_f_calc(f_trat), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(p_trat, gl_trat, gl_res)},
            {"FV": "  Fator A", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(sq_a/gl_a,2), "F Calc": formatar_f_calc((sq_a/gl_a)/qm_res if qm_res>0 else 0), "F Tab": calcular_f_tab(gl_a, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_a/gl_a)/qm_res if qm_res>0 else 0, gl_a, gl_res), gl_a, gl_res)},
            {"FV": "  Fator B", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(sq_b/gl_b,2), "F Calc": formatar_f_calc((sq_b/gl_b)/qm_res if qm_res>0 else 0), "F Tab": calcular_f_tab(gl_b, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_b/gl_b)/qm_res if qm_res>0 else 0, gl_b, gl_res), gl_b, gl_res)},
            {"FV": "  Interação AxB", "GL": gl_inter, "SQ": round(sq_inter,2), "QM": round(sq_inter/gl_inter,2), "F Calc": formatar_f_calc((sq_inter/gl_inter)/qm_res if qm_res>0 else 0), "F Tab": calcular_f_tab(gl_inter, gl_res), "Sig": determinar_sig_texto(p_inter, gl_inter, gl_res)},
            {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(sq_bloc/gl_bloc,2), "F Calc": formatar_f_calc((sq_bloc/gl_bloc)/qm_res if qm_res>0 else 0), "F Tab": calcular_f_tab(gl_bloc, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_bloc/gl_bloc)/qm_res if qm_res>0 else 0, gl_bloc, gl_res), gl_bloc, gl_res)},
            {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
        ]
        
        cv_val = round((np.sqrt(qm_res)/mg)*100, 2) if qm_res > 0 else 0
        cv_string = f"{cv_val}% ({classificar_cv(cv_val)})"

        if tipo_teste == "anova":
            return {"status": "sucesso", "cv": cv_string, "anova": anova}

        # === POST-HOC MÓDULO 2 ===
        fator = 'A' if tipo_teste.endswith("_a") else 'B'
        n_niveis = nA if fator == 'A' else nB
        ep = np.sqrt(qm_res/(nB*nR)) if fator == 'A' else np.sqrt(qm_res/(nA*nR))
        medias = df.groupby(fator)['Valor'].mean().sort_values(ascending=False)

        f_calc_fator = (sq_a/gl_a)/qm_res if fator == 'A' else (sq_b/gl_b)/qm_res
        gl_num_fator = gl_a if fator == 'A' else gl_b
        p_fator = 1 - stats.f.cdf(f_calc_fator, gl_num_fator, gl_res)
        alpha = 0.01 if p_fator < 0.01 else 0.05
        alpha_txt = "1%" if p_fator < 0.01 else "5%"

        if "regr_" in tipo_teste:
            if pd.to_numeric(df[fator], errors='coerce').isna().any(): 
                return {"status": "erro", "mensagem": "A regressão exige doses numéricas quantitativas. Para dados de texto, use Testes de Médias."}
            rep_fator = nB * nR if fator == 'A' else nA * nR
            x, y = np.array(medias.index, dtype=float), np.array(medias.values, dtype=float)
            y_mean = np.mean(y)
            fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(x, y, color='#0f172a', edgecolor='white', linewidth=1.2, label='Médias', zorder=5)
            x_plot = np.linspace(min(x), max(x), 100)
            
            x_otimo, y_otimo = None, None
            if modelo_regr == "linear":
                s, i, r, _, _ = stats.linregress(x, y); y_pred = i + s * x
                ax.plot(x_plot, i + s * x_plot, color='#2d7244', linewidth=3, label='Linear', zorder=4)
                eq, r2, gl_reg = f"y = {i:.4f} {'+' if s>=0 else '-'} {abs(s):.4f}x", r**2, 1
            elif modelo_regr == "quadratica":
                c = np.polyfit(x, y, 2); p = np.poly1d(c); y_pred = p(x)
                ax.plot(x_plot, p(x_plot), color='#2d7244', linewidth=3, label='Quadrática', zorder=4)
                eq, r2, gl_reg = f"y = {c[2]:.4f} {'+' if c[1]>=0 else '-'} {abs(c[1]):.4f}x {'+' if c[0]>=0 else '-'} {abs(c[0]):.4f}x²", 1 - (np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)), 2
                if c[0] != 0:
                    x_otimo = -c[1] / (2 * c[0]); y_otimo = p(x_otimo)
                    ax.axvline(x=x_otimo, color='#94a3b8', linestyle=(0, (4, 4)), alpha=0.85)
                    ax.scatter(x_otimo, y_otimo, color='#b18356', marker='*', s=220, edgecolor='white', linewidth=1.0, zorder=6, label=f'Dose Ótima: {x_otimo:.2f}')

            sq_reg_raw = rep_fator * np.sum((y_pred - y_mean)**2); sq_trat_raw = sq_a if fator == 'A' else sq_b
            sq_desvio_raw = max(0, sq_trat_raw - sq_reg_raw)
            gl_fator_real = gl_a if fator == 'A' else gl_b
            gl_desvio = gl_fator_real - gl_reg
            
            qm_reg = sq_reg_raw / gl_reg if gl_reg > 0 else 0
            qm_desvio = sq_desvio_raw / gl_desvio if gl_desvio > 0 else 0
            f_reg = qm_reg / qm_res if qm_res > 0 else 0
            f_desvio = qm_desvio / qm_res if qm_res > 0 else 0
            p_reg = 1 - stats.f.cdf(f_reg, gl_reg, gl_res) if gl_reg > 0 else np.nan
            p_desvio = 1 - stats.f.cdf(f_desvio, gl_desvio, gl_res) if gl_desvio > 0 else np.nan

            anova_reg = [
                {"FV": f"Regressão ({modelo_regr.capitalize()})", "GL": gl_reg, "SQ": round(sq_reg_raw, 2), "QM": round(qm_reg, 2) if gl_reg>0 else "-", "F Calc": formatar_f_calc(f_reg) if gl_reg>0 else "-", "F Tab": calcular_f_tab(gl_reg, gl_res), "Sig": determinar_sig_texto(p_reg, gl_reg, gl_res)},
                {"FV": "Desvios da Regressão", "GL": gl_desvio, "SQ": round(sq_desvio_raw, 2), "QM": round(qm_desvio, 2) if gl_desvio>0 else "-", "F Calc": formatar_f_calc(f_desvio) if gl_desvio>0 else "-", "F Tab": calcular_f_tab(gl_desvio, gl_res), "Sig": determinar_sig_texto(p_desvio, gl_desvio, gl_res)},
                {"FV": f"Total do Fator {fator}", "GL": gl_fator_real, "SQ": round(sq_trat_raw, 2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]
            ax.set_title(f'Ajuste {modelo_regr.capitalize()} - Fator {fator}', fontsize=15, fontweight='bold', color='#0f172a'); ax.grid(True, color='#e2e8f0', linewidth=1); ax.set_facecolor('#fbfdfb'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
            b_png, b_pdf, b_svg = io.BytesIO(), io.BytesIO(), io.BytesIO()
            fig.savefig(b_png, format="png", bbox_inches='tight')
            fig.savefig(b_pdf, format="pdf", bbox_inches='tight')
            fig.savefig(b_svg, format="svg", bbox_inches='tight')
            plt.close(fig)
            
            res_dict = {"status": "sucesso", "tipo": "regressao", "equacao": eq, "r2": f"{r2:.4f}", "modelo": modelo_regr.capitalize(), "img_png": base64.b64encode(b_png.getvalue()).decode('utf-8'), "img_pdf": base64.b64encode(b_pdf.getvalue()).decode('utf-8'), "img_svg": base64.b64encode(b_svg.getvalue()).decode('utf-8'), "anova_reg": anova_reg}
            if x_otimo is not None: res_dict.update({"dose_otima": f"{x_otimo:.2f}", "resposta_otima": f"{y_otimo:.2f}"})
            return res_dict

        if "tukey_" in tipo_teste:
            q_crit = qsturng(1 - alpha, n_niveis, gl_res); dms = q_crit * ep
            return {"status": "sucesso", "tipo": "tukey", "nome_teste": "Tukey", "alpha_txt": alpha_txt, "q": round(q_crit, 4), "dms": round(dms, 4), "tabela": aplicar_letras(medias, dms_val=dms)}
        
        if "duncan_" in tipo_teste:
            rp_duncan = {p: round(qsturng((1 - alpha)**(p-1), p, gl_res) * ep, 4) for p in range(2, n_niveis + 1)}
            return {"status": "sucesso", "tipo": "duncan", "nome_teste": "Duncan", "alpha_txt": alpha_txt, "alcances": rp_duncan, "tabela": aplicar_letras(medias, duncan_dict=rp_duncan)}

        if "dunnett_" in tipo_teste:
            if not testemunha or testemunha not in medias.index: return {"status": "erro", "mensagem": f"Testemunha '{testemunha}' não encontrada."}
            media_test = medias[testemunha]
            dms_dunnett = stats.t.ppf(1 - (alpha / (2 * (n_niveis - 1))), gl_res) * ep * np.sqrt(2)
            res = [{"Tratamento": str(n), "Média": round(v, 2), "Diferença": round(v - media_test, 2), "Sig": "Significativo (*)" if abs(v - media_test) >= dms_dunnett else "ns"} for n, v in medias.items() if n != testemunha]
            return {"status": "sucesso", "tipo": "dunnett", "alpha_txt": alpha_txt, "dms": round(dms_dunnett, 4), "testemunha": testemunha, "media_testemunha": round(media_test, 2), "resultados": res}

    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ================= MÓDULO 3: PARCELAS SUBDIVIDIDAS =================
@app.post("/api/analise/parcelas")
async def analisar_parcelas(file: UploadFile = File(...), tipo_teste: str = Form("anova"), modelo_regr: str = Form("linear"), testemunha: str = Form("")):
    try:
        content = await file.read()
        try: df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
        cols = list(df.columns)
        df.rename(columns={cols[0]: 'A', cols[1]: 'B', cols[2]: 'Bloco', cols[3]: 'Valor'}, inplace=True)
        df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
        
        nA, nB, nR = len(df['A'].unique()), len(df['B'].unique()), len(df['Bloco'].unique())
        mg, n_total = df['Valor'].mean(), len(df)
        
        sq_tot = ((df['Valor'] - mg)**2).sum()
        sq_bloc = (nA * nB) * ((df.groupby('Bloco')['Valor'].mean() - mg)**2).sum()
        sq_a = (nB * nR) * ((df.groupby('A')['Valor'].mean() - mg)**2).sum()
        sq_a_bloc = nB * ((df.groupby(['A', 'Bloco'])['Valor'].mean() - mg)**2).sum()
        sq_erro_a = max(0, sq_a_bloc - sq_a - sq_bloc)
        sq_parcela_total = sq_a + sq_bloc + sq_erro_a
        
        sq_b = (nA * nR) * ((df.groupby('B')['Valor'].mean() - mg)**2).sum()
        sq_inter = nR * ((df.groupby(['A', 'B'])['Valor'].mean() - mg)**2).sum() - sq_a - sq_b
        sq_erro_b = max(0, sq_tot - sq_bloc - sq_a - sq_erro_a - sq_b - sq_inter)
        
        gl_a, gl_bloc, gl_erro_a = nA - 1, nR - 1, (nA - 1) * (nR - 1)
        gl_parcela_total = (nA * nR) - 1
        gl_b, gl_inter, gl_tot = nB - 1, (nA - 1) * (nB - 1), n_total - 1
        gl_erro_b = gl_tot - (gl_a + gl_bloc + gl_erro_a + gl_b + gl_inter)
        
        qm_a = sq_a / gl_a
        qm_bloc = sq_bloc / gl_bloc
        qm_erro_a = sq_erro_a / gl_erro_a if gl_erro_a > 0 else 0
        qm_b = sq_b / gl_b
        qm_inter = sq_inter / gl_inter
        qm_erro_b = sq_erro_b / gl_erro_b if gl_erro_b > 0 else 0
        
        f_a = qm_a / qm_erro_a if qm_erro_a > 0 else 0
        f_bloc = qm_bloc / qm_erro_a if qm_erro_a > 0 else 0
        f_b = qm_b / qm_erro_b if qm_erro_b > 0 else 0
        f_inter = qm_inter / qm_erro_b if qm_erro_b > 0 else 0
        p_inter = 1 - stats.f.cdf(f_inter, gl_inter, gl_erro_b)

        anova = [
            {"FV": "Fator A (Parcela)", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(qm_a,2), "F Calc": formatar_f_calc(f_a), "F Tab": calcular_f_tab(gl_a, gl_erro_a), "Sig": determinar_sig_texto(1-stats.f.cdf(f_a, gl_a, gl_erro_a), gl_a, gl_erro_a)},
            {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(qm_bloc,2), "F Calc": formatar_f_calc(f_bloc), "F Tab": calcular_f_tab(gl_bloc, gl_erro_a), "Sig": determinar_sig_texto(1-stats.f.cdf(f_bloc, gl_bloc, gl_erro_a), gl_bloc, gl_erro_a)},
            {"FV": "Erro A (Parcela)", "GL": gl_erro_a, "SQ": round(sq_erro_a,2), "QM": round(qm_erro_a,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Parcelas (Subtotal)", "GL": gl_parcela_total, "SQ": round(sq_parcela_total,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Fator B (Subparcela)", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(qm_b,2), "F Calc": formatar_f_calc(f_b), "F Tab": calcular_f_tab(gl_b, gl_erro_b), "Sig": determinar_sig_texto(1-stats.f.cdf(f_b, gl_b, gl_erro_b), gl_b, gl_erro_b)},
            {"FV": "Interação AxB", "GL": gl_inter, "SQ": round(sq_inter,2), "QM": round(qm_inter,2), "F Calc": formatar_f_calc(f_inter), "F Tab": calcular_f_tab(gl_inter, gl_erro_b), "Sig": determinar_sig_texto(p_inter, gl_inter, gl_erro_b)},
            {"FV": "Erro B (Subparcela)", "GL": gl_erro_b, "SQ": round(sq_erro_b,2), "QM": round(qm_erro_b,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
        ]

        cv_a_val = round((np.sqrt(qm_erro_a)/mg)*100, 2) if qm_erro_a > 0 else 0
        cv_b_val = round((np.sqrt(qm_erro_b)/mg)*100, 2) if qm_erro_b > 0 else 0

        if tipo_teste == "anova":
            return {"status": "sucesso", "cv_a": f"{cv_a_val}% ({classificar_cv(cv_a_val)})", "cv_b": f"{cv_b_val}% ({classificar_cv(cv_b_val)})", "anova": anova}

        # === POST-HOC MÓDULO 3 ===
        fator = 'A' if tipo_teste.endswith("_a") else 'B'
        n_niveis = nA if fator == 'A' else nB
        qm_err_atual = qm_erro_a if fator == 'A' else qm_erro_b
        gl_err_atual = gl_erro_a if fator == 'A' else gl_erro_b
        rep_fator = (nB * nR) if fator == 'A' else (nA * nR)

        if qm_err_atual <= 0 or gl_err_atual <= 0: return {"status": "erro", "mensagem": f"O Erro {fator} é zero ou negativo, impossibilitando testes."}
        ep = np.sqrt(qm_err_atual / rep_fator)
        medias = df.groupby(fator)['Valor'].mean().sort_values(ascending=False)

        f_calc_fator = f_a if fator == 'A' else f_b
        gl_num_fator = gl_a if fator == 'A' else gl_b
        p_fator = 1 - stats.f.cdf(f_calc_fator, gl_num_fator, gl_err_atual)
        alpha = 0.01 if p_fator < 0.01 else 0.05
        alpha_txt = "1%" if p_fator < 0.01 else "5%"

        if "regr_" in tipo_teste:
            if pd.to_numeric(df[fator], errors='coerce').isna().any(): 
                return {"status": "erro", "mensagem": "A regressão exige doses numéricas quantitativas."}
            x, y = np.array(medias.index, dtype=float), np.array(medias.values, dtype=float)
            y_mean = np.mean(y)
            fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(x, y, color='#0f172a', edgecolor='white', linewidth=1.2, label='Médias', zorder=5)
            x_plot = np.linspace(min(x), max(x), 100)
            
            x_otimo, y_otimo = None, None
            if modelo_regr == "linear":
                s, i, r, _, _ = stats.linregress(x, y); y_pred = i + s * x
                ax.plot(x_plot, i + s * x_plot, color='#2d7244', linewidth=3, label='Linear', zorder=4)
                eq, r2, gl_reg = f"y = {i:.4f} {'+' if s>=0 else '-'} {abs(s):.4f}x", r**2, 1
            elif modelo_regr == "quadratica":
                c = np.polyfit(x, y, 2); p = np.poly1d(c); y_pred = p(x)
                ax.plot(x_plot, p(x_plot), color='#2d7244', linewidth=3, label='Quadrática', zorder=4)
                eq, r2, gl_reg = f"y = {c[2]:.4f} {'+' if c[1]>=0 else '-'} {abs(c[1]):.4f}x {'+' if c[0]>=0 else '-'} {abs(c[0]):.4f}x²", 1 - (np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)), 2
                if c[0] != 0:
                    x_otimo = -c[1] / (2 * c[0]); y_otimo = p(x_otimo)
                    ax.axvline(x=x_otimo, color='#94a3b8', linestyle=(0, (4, 4)), alpha=0.85)
                    ax.scatter(x_otimo, y_otimo, color='#b18356', marker='*', s=220, edgecolor='white', linewidth=1.0, zorder=6, label=f'Dose Ótima: {x_otimo:.2f}')

            sq_reg_raw = rep_fator * np.sum((y_pred - y_mean)**2); sq_trat_raw = sq_a if fator == 'A' else sq_b
            sq_desvio_raw = max(0, sq_trat_raw - sq_reg_raw)
            gl_fator_real = n_niveis - 1; gl_desvio = gl_fator_real - gl_reg
            qm_reg = sq_reg_raw / gl_reg if gl_reg > 0 else 0
            qm_desvio = sq_desvio_raw / gl_desvio if gl_desvio > 0 else 0
            f_reg = qm_reg / qm_err_atual if qm_err_atual > 0 else 0
            f_desvio = qm_desvio / qm_err_atual if qm_err_atual > 0 else 0
            p_reg = 1 - stats.f.cdf(f_reg, gl_reg, gl_err_atual) if gl_reg > 0 else np.nan
            p_desvio = 1 - stats.f.cdf(f_desvio, gl_desvio, gl_err_atual) if gl_desvio > 0 else np.nan

            anova_reg = [
                {"FV": f"Regressão ({modelo_regr.capitalize()})", "GL": gl_reg, "SQ": round(sq_reg_raw, 2), "QM": round(qm_reg, 2) if gl_reg>0 else "-", "F Calc": formatar_f_calc(f_reg) if gl_reg>0 else "-", "F Tab": calcular_f_tab(gl_reg, gl_err_atual), "Sig": determinar_sig_texto(p_reg, gl_reg, gl_err_atual)},
                {"FV": "Desvios da Regressão", "GL": gl_desvio, "SQ": round(sq_desvio_raw, 2), "QM": round(qm_desvio, 2) if gl_desvio>0 else "-", "F Calc": formatar_f_calc(f_desvio) if gl_desvio>0 else "-", "F Tab": calcular_f_tab(gl_desvio, gl_err_atual), "Sig": determinar_sig_texto(p_desvio, gl_desvio, gl_err_atual)},
                {"FV": f"Total do Fator {fator}", "GL": gl_fator_real, "SQ": round(sq_trat_raw, 2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]
            ax.set_title(f'Ajuste de regressão - Fator {fator}', fontsize=15, fontweight='bold', color='#0f172a'); ax.grid(True, color='#e2e8f0', linewidth=1); ax.set_facecolor('#fbfdfb'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
            b_png, b_pdf = io.BytesIO(), io.BytesIO()
            fig.savefig(b_png, format="png"); fig.savefig(b_pdf, format="pdf"); plt.close(fig)
            res_dict = {"status": "sucesso", "tipo": "regressao", "equacao": eq, "r2": f"{r2:.4f}", "modelo": modelo_regr.capitalize(), "img_png": base64.b64encode(b_png.getvalue()).decode('utf-8'), "img_pdf": base64.b64encode(b_pdf.getvalue()).decode('utf-8'), "img_svg": base64.b64encode(b_svg.getvalue()).decode('utf-8'), "anova_reg": anova_reg}
            if x_otimo is not None: res_dict.update({"dose_otima": f"{x_otimo:.2f}", "resposta_otima": f"{y_otimo:.2f}"})
            return res_dict

        if "tukey_" in tipo_teste:
            q_crit = qsturng(1 - alpha, n_niveis, gl_err_atual); dms = q_crit * ep
            return {"status": "sucesso", "tipo": "tukey", "nome_teste": "Tukey", "alpha_txt": alpha_txt, "q": round(q_crit, 4), "dms": round(dms, 4), "tabela": aplicar_letras(medias, dms_val=dms)}
        
        if "duncan_" in tipo_teste:
            rp_duncan = {p: round(qsturng((1 - alpha)**(p-1), p, gl_err_atual) * ep, 4) for p in range(2, n_niveis + 1)}
            return {"status": "sucesso", "tipo": "duncan", "nome_teste": "Duncan", "alpha_txt": alpha_txt, "alcances": rp_duncan, "tabela": aplicar_letras(medias, duncan_dict=rp_duncan)}

        if "dunnett_" in tipo_teste:
            if not testemunha or testemunha not in medias.index: return {"status": "erro", "mensagem": f"Testemunha '{testemunha}' não encontrada."}
            media_test = medias[testemunha]
            dms_dunnett = stats.t.ppf(1 - (alpha / (2 * (n_niveis - 1))), gl_err_atual) * ep * np.sqrt(2)
            res = [{"Tratamento": str(n), "Média": round(v, 2), "Diferença": round(v - media_test, 2), "Sig": "Significativo (*)" if abs(v - media_test) >= dms_dunnett else "ns"} for n, v in medias.items() if n != testemunha]
            return {"status": "sucesso", "tipo": "dunnett", "alpha_txt": alpha_txt, "dms": round(dms_dunnett, 4), "testemunha": testemunha, "media_testemunha": round(media_test, 2), "resultados": res}

    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


# ================= MÓDULO ADICIONAL: REGRESSÃO DIRETA DOSE x PRODUTIVIDADE =================
def _parse_lista_numerica(texto: str):
    if not texto:
        return []
    import re
    texto = texto.strip()
    # Preferimos separar por linha, ponto e vírgula ou tabulação para preservar decimal com vírgula (ex.: 16,3).
    if re.search(r'[;\n\r\t]', texto):
        partes = [x for x in re.split(r'[;\n\r\t]+', texto) if x.strip()]
    else:
        partes = [x for x in re.split(r'\s+', texto) if x.strip()]
    return [float(x.strip().replace(',', '.')) for x in partes]

def _modelo_nome(modelo_regr: str):
    m = str(modelo_regr or "quadratica").lower().strip()
    aliases = {
        "linear": "linear",
        "quadratica": "quadratica", "quadrática": "quadratica",
        "cubica": "cubica", "cúbica": "cubica",
        "exponencial": "exponencial", "exp": "exponencial",
        "logaritmica": "logaritmica", "logarítmica": "logaritmica", "log": "logaritmica"
    }
    return aliases.get(m, "quadratica")

def _fmt2(v):
    return f"{float(v):.2f}"

def _ajustar_regressao_direta(x, y, modelo_regr="quadratica"):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Informe a mesma quantidade de doses e produtividades, com pelo menos 2 pares.")
    ordem = np.argsort(x)
    x, y = x[ordem], y[ordem]
    modelo = _modelo_nome(modelo_regr)

    if modelo in ["linear", "quadratica", "cubica"]:
        grau = {"linear": 1, "quadratica": 2, "cubica": 3}[modelo]
        if len(x) <= grau:
            raise ValueError(f"O modelo {modelo} exige pelo menos {grau + 1} pontos.")
        coef = np.polyfit(x, y, grau)
        pol = np.poly1d(coef)
        y_pred = pol(x)
        pred_func = lambda z: np.poly1d(coef)(z)
        if grau == 1:
            eq = f"y = {coef[1]:.4f} {'+' if coef[0] >= 0 else '-'} {abs(coef[0]):.4f}x"
            modelo_nome = "Linear"
        elif grau == 2:
            eq = f"y = {coef[2]:.4f} {'+' if coef[1] >= 0 else '-'} {abs(coef[1]):.4f}x {'+' if coef[0] >= 0 else '-'} {abs(coef[0]):.4f}x²"
            modelo_nome = "Quadrática"
        else:
            eq = f"y = {coef[3]:.4f} {'+' if coef[2] >= 0 else '-'} {abs(coef[2]):.4f}x {'+' if coef[1] >= 0 else '-'} {abs(coef[1]):.4f}x² {'+' if coef[0] >= 0 else '-'} {abs(coef[0]):.4f}x³"
            modelo_nome = "Cúbica"
    elif modelo == "exponencial":
        if np.any(y <= 0):
            raise ValueError("A regressão exponencial exige produtividades maiores que zero.")
        b, loga = np.polyfit(x, np.log(y), 1)
        a = float(np.exp(loga))
        y_pred = a * np.exp(b * x)
        pred_func = lambda z: a * np.exp(b * np.array(z))
        eq = f"y = {a:.4f}e^({b:.4f}x)"
        modelo_nome = "Exponencial"
        coef = np.array([a, b])
    elif modelo == "logaritmica":
        if np.any(x <= 0):
            raise ValueError("A regressão logarítmica exige doses maiores que zero. Evite dose 0 ou use outro modelo.")
        b, a = np.polyfit(np.log(x), y, 1)
        y_pred = a + b * np.log(x)
        pred_func = lambda z: a + b * np.log(np.array(z))
        eq = f"y = {a:.4f} {'+' if b >= 0 else '-'} {abs(b):.4f}ln(x)"
        modelo_nome = "Logarítmica"
        coef = np.array([a, b])
    else:
        raise ValueError("Modelo de regressão inválido.")

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    x_min, x_max = float(np.min(x)), float(np.max(x))
    dx = (x_max - x_min) or 1.0
    x_plot = np.linspace(x_min, x_max, 500)
    y_plot = np.asarray(pred_func(x_plot), dtype=float)

    # Dose ótima: máximo estimado dentro do intervalo observado, válido também para cúbica,
    # exponencial e logarítmica. Para modelos monotônicos, o ótimo será uma das extremidades.
    idx_otimo = int(np.nanargmax(y_plot))
    x_otimo = float(x_plot[idx_otimo])
    y_otimo = float(y_plot[idx_otimo])

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fbfdfb')
    ax.plot(x_plot, y_plot, color='#2d7244', linewidth=3.0, solid_capstyle='round', label='Curva ajustada', zorder=3)
    ax.scatter(x, y, s=78, color='#0f172a', edgecolor='white', linewidth=1.5, label='Dados observados', zorder=4)

    y_all = np.r_[y, y_plot, y_otimo]
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    dy = (y_max - y_min) or 1.0
    ax.set_xlim(x_min - dx * 0.05, x_max + dx * 0.05)
    ax.set_ylim(y_min - dy * 0.16, y_max + dy * 0.18)

    ax.axvline(x=x_otimo, color='#94a3b8', linestyle=(0, (4, 4)), linewidth=1.4, zorder=1)
    ax.scatter(x_otimo, y_otimo, color='#b18356', marker='*', s=260, edgecolor='white', linewidth=1.1, zorder=5)
    ax.annotate(
        f"Dose ótima: {_fmt2(x_otimo)}\nProd. ótima: {_fmt2(y_otimo)}",
        xy=(x_otimo, y_otimo), xytext=(14, 18), textcoords='offset points',
        fontsize=10.5, fontweight='bold', color='#1b3c28',
        bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#dcf1e1', lw=1.2, alpha=0.96),
        arrowprops=dict(arrowstyle='->', color='#2d7244', lw=1.2), zorder=6
    )
    ax.set_title('Regressão direta para dose ótima', fontsize=15, fontweight='bold', color='#0f172a', pad=14)
    ax.set_xlabel('Dose', fontsize=11.5, fontweight='bold', color='#334155')
    ax.set_ylabel('Produtividade', fontsize=11.5, fontweight='bold', color='#334155')
    ax.grid(True, which='major', color='#e2e8f0', linewidth=1, linestyle='-')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1'); ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=10)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.96, fontsize=10)
    fig.tight_layout()

    b_png, b_pdf, b_svg = io.BytesIO(), io.BytesIO(), io.BytesIO()
    fig.savefig(b_png, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(b_pdf, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(b_svg, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "status": "sucesso", "tipo": "regressao_direta", "modelo": modelo_nome,
        "equacao": eq, "r2": f"{r2:.4f}",
        "dose_otima": _fmt2(x_otimo), "resposta_otima": _fmt2(y_otimo), "produtividade_otima": _fmt2(y_otimo),
        "img_png": base64.b64encode(b_png.getvalue()).decode('utf-8'),
        "img_pdf": base64.b64encode(b_pdf.getvalue()).decode('utf-8'),
        "img_svg": base64.b64encode(b_svg.getvalue()).decode('utf-8'),
        "pontos": [{"Dose": round(float(a), 4), "Produtividade": round(float(b), 4), "Estimado": round(float(c), 4)} for a, b, c in zip(x, y, y_pred)]
    }

@app.post("/api/analise/regressao-direta")
async def regressao_direta(file: UploadFile = File(None), doses: str = Form(""), produtividades: str = Form(""), modelo_regr: str = Form("quadratica")):
    try:
        if file is not None:
            content = await file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
            if df.shape[1] < 2:
                return {"status": "erro", "mensagem": "O CSV precisa ter ao menos duas colunas: Dose e Produtividade."}
            x = pd.to_numeric(df.iloc[:, 0].astype(str).str.replace(',', '.'), errors='coerce')
            y = pd.to_numeric(df.iloc[:, 1].astype(str).str.replace(',', '.'), errors='coerce')
            dados = pd.DataFrame({'Dose': x, 'Produtividade': y}).dropna()
            return _ajustar_regressao_direta(dados['Dose'].values, dados['Produtividade'].values, modelo_regr)

        x = _parse_lista_numerica(doses)
        y = _parse_lista_numerica(produtividades)
        return _ajustar_regressao_direta(x, y, modelo_regr)
    except ValueError as e:
        return {"status": "erro", "mensagem": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
