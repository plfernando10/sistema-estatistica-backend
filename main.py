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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def determinar_sig_texto(p_valor, gl_num, gl_den):
    if pd.isna(p_valor) or gl_den <= 0: return "ns (p >= 0,05)"
    if p_valor < 0.01: return "1% (**)"
    elif p_valor < 0.05: return "5% (*)"
    else: return "ns (p >= 0,05)"

@app.post("/api/analise/fatorial")
async def analisar_fatorial(file: UploadFile = File(...), tipo_teste: str = Form("anova"), modelo_regr: str = Form("linear")):
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

        # === ANOVA ===
        if tipo_teste == "anova":
            return {"status": "sucesso", "cv": round((np.sqrt(qm_res)/mg)*100, 2), "configuracao": f"A({nA}), B({nB}), Blocos({nR})",
                    "anova": [
                        {"FV": "Fator A", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(sq_a/gl_a,2), "F Calc": round((sq_a/gl_a)/qm_res,2), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_a/gl_a)/qm_res, gl_a, gl_res), gl_a, gl_res)},
                        {"FV": "Fator B", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(sq_b/gl_b,2), "F Calc": round((sq_b/gl_b)/qm_res,2), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_b/gl_b)/qm_res, gl_b, gl_res), gl_b, gl_res)},
                        {"FV": "Interação", "GL": (nA-1)*(nB-1), "SQ": round(sq_inter,2), "QM": round(sq_inter/((nA-1)*(nB-1)),2), "F Calc": round((sq_inter/((nA-1)*(nB-1)))/qm_res,2), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_inter/((nA-1)*(nB-1)))/qm_res, (nA-1)*(nB-1), gl_res), (nA-1)*(nB-1), gl_res)},
                        {"FV": "Blocos", "GL": nR-1, "SQ": round(sq_bloc,2), "QM": round(sq_bloc/(nR-1),2), "F Calc": "-", "Sig": "-"},
                        {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "Sig": "-"}
                    ]}
        
        # === REGRESSÃO COM TRAVA E MÚLTIPLOS MODELOS ===
        if "regr_" in tipo_teste:
            fator = 'A' if tipo_teste == "regr_a" else 'B'
            
            # Trava 1: Erro de String (Ex: "C1")
            if pd.to_numeric(df[fator], errors='coerce').isna().any():
                return {"status": "erro", "mensagem": f"O Fator {fator} possui tratamentos qualitativos (texto). A regressão exige doses numéricas. Utilize o Teste de Tukey."}
            
            medias_dose = df.groupby(fator)['Valor'].mean()
            x = np.array(medias_dose.index, dtype=float)
            y = np.array(medias_dose.values, dtype=float)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, color='black', label='Médias Observadas')
            x_plot = np.linspace(min(x), max(x), 100)
            eq_texto, r2_texto = "", ""
            
            try:
                if modelo_regr == "linear":
                    slope, intercept, r_val, _, _ = stats.linregress(x, y)
                    ax.plot(x_plot, intercept + slope * x_plot, color='blue', label='Ajuste Linear')
                    eq_texto, r2_texto = f"y = {intercept:.4f} + {slope:.4f}x", f"{r_val**2:.4f}"
                
                elif modelo_regr == "quadratica":
                    coeffs = np.polyfit(x, y, 2); p_quad = np.poly1d(coeffs)
                    ax.plot(x_plot, p_quad(x_plot), color='red', label='Ajuste Quadrático')
                    eq_texto = f"y = {coeffs[2]:.4f} + ({coeffs[1]:.4f})x + ({coeffs[0]:.4f})x²"
                    r2_texto = f"{1 - (np.sum((y - p_quad(x))**2) / np.sum((y - np.mean(y))**2)):.4f}"
                    if coeffs[0] != 0:
                        x_o = -coeffs[1] / (2 * coeffs[0])
                        if min(x) <= x_o <= max(x): ax.plot(x_o, p_quad(x_o), 'go', markersize=10, label=f'Ótimo: {x_o:.2f}')
                
                elif modelo_regr == "logaritmica":
                    if any(x <= 0): return {"status": "erro", "mensagem": "Existem doses <= 0. Não é possível calcular logaritmo de zero."}
                    slope, intercept, r_val, _, _ = stats.linregress(np.log(x), y)
                    ax.plot(x_plot, intercept + slope * np.log(x_plot), color='green', label='Ajuste Logarítmico')
                    eq_texto, r2_texto = f"y = {intercept:.4f} + {slope:.4f} * ln(x)", f"{r_val**2:.4f}"
                    
                elif modelo_regr == "exponencial":
                    if any(y <= 0): return {"status": "erro", "mensagem": "Respostas <= 0 não permitem ajuste exponencial."}
                    slope, intercept, r_val, _, _ = stats.linregress(x, np.log(y))
                    ax.plot(x_plot, np.exp(intercept) * np.exp(slope * x_plot), color='purple', label='Ajuste Exponencial')
                    eq_texto, r2_texto = f"y = {np.exp(intercept):.4f} * e^({slope:.4f}x)", f"{r_val**2:.4f}"
            except Exception as e:
                return {"status": "erro", "mensagem": f"Erro ao ajustar o modelo matemático: {e}"}

            ax.set_title(f"Regressão {modelo_regr.capitalize()} - Fator {fator}"); ax.set_xlabel("Doses"); ax.set_ylabel("Resposta"); ax.legend(); ax.grid(True)
            buf_png, buf_pdf = io.BytesIO(), io.BytesIO()
            fig.savefig(buf_png, format="png", bbox_inches='tight'); fig.savefig(buf_pdf, format="pdf", bbox_inches='tight'); plt.close(fig)
            
            return {"status": "sucesso", "tipo": "regressao", "equacao": eq_texto, "r2": r2_texto, "modelo": modelo_regr.capitalize(),
                    "img_png": base64.b64encode(buf_png.getvalue()).decode('utf-8'), "img_pdf": base64.b64encode(buf_pdf.getvalue()).decode('utf-8')}
            
        # === TUKEY ===
        if "tukey_" in tipo_teste:
            fator = 'A' if tipo_teste == "tukey_a" else 'B'
            n_niveis = nA if fator == 'A' else nB
            erro_padrao = np.sqrt(qm_res/(nB*nR)) if fator == 'A' else np.sqrt(qm_res/(nA*nR))
            medias = df.groupby(fator)['Valor'].mean().sort_values(ascending=False)
            q_crit = qsturng(0.95, n_niveis, gl_res); dms = q_crit * erro_padrao
            
            # Agrupamento simplificado para visualização Web
            resultado = [{"Nível": str(n), "Média": round(v, 2), "Letra": "a"} for n, v in medias.items()]
            return {"status": "sucesso", "tipo": "tukey", "q": round(q_crit, 4), "dms": round(dms, 4), "tabela": resultado}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
