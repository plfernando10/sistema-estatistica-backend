from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.libqsturng import qsturng
import matplotlib
matplotlib.use('Agg') # Fundamental para rodar gráficos em nuvem sem travar
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def determinar_sig_texto(p_valor, gl_num, gl_den):
    if pd.isna(p_valor) or gl_den <= 0: return "ns (p >= 0,05)"
    if p_valor < 0.01: return "1% (**)"
    elif p_valor < 0.05: return "5% (*)"
    else: return "ns (p >= 0,05)"

def gerar_grafico_base64(x, y, p_quad, titulo, coeffs):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color='black', label='Médias')
    x_plot = np.linspace(min(x), max(x), 100)
    ax.plot(x_plot, p_quad(x_plot), color='red', label='Ajuste Quadrático')
    if coeffs[0] != 0:
        x_o = -coeffs[1] / (2 * coeffs[0])
        if min(x) <= x_o <= max(x): ax.plot(x_o, p_quad(x_o), 'go', markersize=10, label=f'Ótimo: {x_o:.2f}')
    ax.set_title(titulo); ax.set_xlabel("Doses"); ax.set_ylabel("Resposta")
    ax.legend(); ax.grid(True)
    
    buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", bbox_inches='tight')
    buf_pdf = io.BytesIO(); fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf_png.getvalue()).decode('utf-8'), base64.b64encode(buf_pdf.getvalue()).decode('utf-8')

@app.post("/api/analise/fatorial")
async def analisar_fatorial(file: UploadFile = File(...), tipo_teste: str = Form("anova")):
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
        sq_res = max(0, sq_tot - sq_bloc - sq_a - sq_b - sq_inter)
        
        gl_a, gl_b, gl_res = nA-1, nB-1, (nA*nB*nR)-1 - (nA-1) - (nB-1) - ((nA-1)*(nB-1)) - (nR-1)
        qm_res = sq_res/gl_res if gl_res > 0 else 0

        # === SE FOR PEDIDO APENAS A ANOVA ===
        if tipo_teste == "anova":
            return {"status": "sucesso", "cv": round((np.sqrt(qm_res)/mg)*100, 2), "configuracao": f"A({nA}), B({nB}), Blocos({nR})",
                    "anova": [
                        {"FV": "Fator A", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(sq_a/gl_a,2), "F Calc": round((sq_a/gl_a)/qm_res,2), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_a/gl_a)/qm_res, gl_a, gl_res), gl_a, gl_res)},
                        {"FV": "Fator B", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(sq_b/gl_b,2), "F Calc": round((sq_b/gl_b)/qm_res,2), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_b/gl_b)/qm_res, gl_b, gl_res), gl_b, gl_res)},
                        {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "Sig": "-"}
                    ]}
        
        # === SE FOR PEDIDO REGRESSÃO ===
        if "regr_" in tipo_teste:
            fator = 'A' if tipo_teste == "regr_a" else 'B'
            medias_dose = df.groupby(fator)['Valor'].mean()
            x = np.array(medias_dose.index, dtype=float)
            y = np.array(medias_dose.values, dtype=float)
            slope, intercept, r_val, _, _ = stats.linregress(x, y)
            coeffs = np.polyfit(x, y, 2); p_quad = np.poly1d(coeffs)
            r2_q = 1 - (np.sum((y - p_quad(x))**2) / np.sum((y - np.mean(y))**2))
            png, pdf = gerar_grafico_base64(x, y, p_quad, f"Regressão Fator {fator}", coeffs)
            
            return {
                "tipo": "regressao",
                "linear": f"y = {intercept:.2f} + {slope:.2f}x (R²: {r_val**2:.2f})",
                "quad": f"y = {coeffs[2]:.2f} + ({coeffs[1]:.2f})x + ({coeffs[0]:.2f})x² (R²: {r2_q:.2f})",
                "img_png": png, "img_pdf": pdf
            }
            
        # === SE FOR PEDIDO TUKEY ===
        if "tukey_" in tipo_teste:
            fator = 'A' if tipo_teste == "tukey_a" else 'B'
            n_niveis = nA if fator == 'A' else nB
            erro_padrao = np.sqrt(qm_res/(nB*nR)) if fator == 'A' else np.sqrt(qm_res/(nA*nR))
            medias = df.groupby(fator)['Valor'].mean().sort_values(ascending=False)
            
            q_crit = qsturng(0.95, n_niveis, gl_res)
            dms = q_crit * erro_padrao
            
            # Lógica simples de agrupamento para o frontend (pode ser refinada depois)
            resultado_letras = []
            letra_atual = 'a'
            for nome, val in medias.items():
                resultado_letras.append({"Nível": str(nome), "Média": round(val, 2), "Letra": letra_atual})
            
            return {
                "tipo": "tukey",
                "q": round(q_crit, 4), "dms": round(dms, 4),
                "tabela": resultado_letras
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
