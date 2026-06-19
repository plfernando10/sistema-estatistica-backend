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
    if pd.isna(p_valor) or gl_den <= 0 or p_valor > 1: return "ns"
    if p_valor < 0.01: return "1% (**)"
    elif p_valor < 0.05: return "5% (*)"
    else: return "ns"

def calcular_f_tab(gl_num, gl_den, alpha=0.95):
    if gl_num > 0 and gl_den > 0: 
        return round(stats.f.ppf(alpha, gl_num, gl_den), 2)
    return "-"

# ================= MÓDULO 1: SIMPLES =================
@app.post("/api/analise/simples")
async def analisar_simples(file: UploadFile = File(...), tipo_delineamento: str = Form("dbc")):
    try:
        content = await file.read()
        try: 
            df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: 
            df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
        
        cols = list(df.columns)
        if tipo_delineamento == "dql":
            df.rename(columns={cols[0]: 'Trat', cols[1]: 'Linha', cols[2]: 'Coluna', cols[3]: 'Valor'}, inplace=True)
        else:
            df.rename(columns={cols[0]: 'Trat', cols[1]: 'Bloco', cols[2]: 'Valor'}, inplace=True)
            
        df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
        mg = df['Valor'].mean()
        sq_tot = ((df['Valor'] - mg)**2).sum()
        n_total = len(df)
        
        if tipo_delineamento == "dql":
            nT = len(df['Trat'].unique())
            sq_trat = nT * ((df.groupby('Trat')['Valor'].mean() - mg)**2).sum()
            sq_linha = nT * ((df.groupby('Linha')['Valor'].mean() - mg)**2).sum()
            sq_col = nT * ((df.groupby('Coluna')['Valor'].mean() - mg)**2).sum()
            sq_res = max(0, sq_tot - sq_trat - sq_linha - sq_col)
            gl_trat, gl_linha, gl_col = nT - 1, nT - 1, nT - 1
            gl_res, gl_tot = (nT - 1) * (nT - 2), n_total - 1
            qm_trat, qm_linha, qm_col = sq_trat / gl_trat, sq_linha / gl_linha, sq_col / gl_col
            qm_res = sq_res / gl_res if gl_res > 0 else 0
            f_trat = qm_trat/qm_res if qm_res>0 else 0
            f_linha = qm_linha/qm_res if qm_res>0 else 0
            f_col = qm_col/qm_res if qm_res>0 else 0
            
            anova = [
                {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": round(f_trat,2), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_trat, gl_trat, gl_res), gl_trat, gl_res)},
                {"FV": "Linhas", "GL": gl_linha, "SQ": round(sq_linha,2), "QM": round(qm_linha,2), "F Calc": round(f_linha,2), "F Tab": calcular_f_tab(gl_linha, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_linha, gl_linha, gl_res), gl_linha, gl_res)},
                {"FV": "Colunas", "GL": gl_col, "SQ": round(sq_col,2), "QM": round(qm_col,2), "F Calc": round(f_col,2), "F Tab": calcular_f_tab(gl_col, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_col, gl_col, gl_res), gl_col, gl_res)},
                {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]
        else:
            nT = len(df['Trat'].unique())
            nR = len(df['Bloco'].unique()) if tipo_delineamento == "dbc" else 1
            sq_trat = nR * ((df.groupby('Trat')['Valor'].mean() - mg)**2).sum() if tipo_delineamento == "dbc" else (n_total/nT) * ((df.groupby('Trat')['Valor'].mean() - mg)**2).sum()
            
            if tipo_delineamento == "dbc":
                sq_bloc = nT * ((df.groupby('Bloco')['Valor'].mean() - mg)**2).sum()
                sq_res = max(0, sq_tot - sq_trat - sq_bloc)
                gl_trat, gl_bloc, gl_tot = nT - 1, nR - 1, n_total - 1
                gl_res = gl_tot - gl_trat - gl_bloc
                qm_trat, qm_bloc = sq_trat/gl_trat, sq_bloc/gl_bloc
                qm_res = sq_res/gl_res if gl_res>0 else 0
                f_trat = qm_trat/qm_res if qm_res>0 else 0
                f_bloc = qm_bloc/qm_res if qm_res>0 else 0
                anova = [
                    {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": round(f_trat,2), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_trat, gl_trat, gl_res), gl_trat, gl_res)},
                    {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(qm_bloc,2), "F Calc": round(f_bloc,2), "F Tab": calcular_f_tab(gl_bloc, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_bloc, gl_bloc, gl_res), gl_bloc, gl_res)},
                    {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"}
                ]
            else: 
                sq_res = max(0, sq_tot - sq_trat)
                gl_trat, gl_tot = nT - 1, n_total - 1
                gl_res = gl_tot - gl_trat
                qm_trat = sq_trat/gl_trat
                qm_res = sq_res/gl_res if gl_res>0 else 0
                f_trat = qm_trat/qm_res if qm_res>0 else 0
                anova = [
                    {"FV": "Tratamentos", "GL": gl_trat, "SQ": round(sq_trat,2), "QM": round(qm_trat,2), "F Calc": round(f_trat,2), "F Tab": calcular_f_tab(gl_trat, gl_res), "Sig": determinar_sig_texto(1 - stats.f.cdf(f_trat, gl_trat, gl_res), gl_trat, gl_res)},
                    {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"}
                ]
        cv = (np.sqrt(qm_res)/mg)*100 if qm_res > 0 else 0
        return {"status": "sucesso", "cv": round(cv, 2), "anova": anova}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

# ================= MÓDULO 2: FATORIAL (COM POST-HOC) =================
@app.post("/api/analise/fatorial")
async def analisar_fatorial(file: UploadFile = File(...), tipo_teste: str = Form("anova"), modelo_regr: str = Form("linear"), testemunha: str = Form("")):
    try:
        content = await file.read()
        try: 
            df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: 
            df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
            
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
        
        gl_a, gl_b, gl_inter = nA-1, nB-1, (nA-1)*(nB-1)
        gl_bloc, gl_tot = nR-1, (nA*nB*nR)-1
        gl_res = gl_tot - gl_a - gl_b - gl_inter - gl_bloc
        qm_res = sq_res/gl_res if gl_res > 0 else 0

        if tipo_teste == "anova":
            return {
                "status": "sucesso", 
                "cv": round((np.sqrt(qm_res)/mg)*100, 2), 
                "anova": [
                    {"FV": "Fator A", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(sq_a/gl_a,2), "F Calc": round((sq_a/gl_a)/qm_res,2), "F Tab": calcular_f_tab(gl_a, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_a/gl_a)/qm_res, gl_a, gl_res), gl_a, gl_res)},
                    {"FV": "Fator B", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(sq_b/gl_b,2), "F Calc": round((sq_b/gl_b)/qm_res,2), "F Tab": calcular_f_tab(gl_b, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_b/gl_b)/qm_res, gl_b, gl_res), gl_b, gl_res)},
                    {"FV": "Interação", "GL": gl_inter, "SQ": round(sq_inter,2), "QM": round(sq_inter/gl_inter,2), "F Calc": round((sq_inter/gl_inter)/qm_res,2), "F Tab": calcular_f_tab(gl_inter, gl_res), "Sig": determinar_sig_texto(1-stats.f.cdf((sq_inter/gl_inter)/qm_res, gl_inter, gl_res), gl_inter, gl_res)},
                    {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "F Tab": "-", "Sig": "-"}
                ]
            }
        
        fator = 'A' if tipo_teste.endswith("_a") else 'B'
        n_niveis = nA if fator == 'A' else nB
        ep = np.sqrt(qm_res/(nB*nR)) if fator == 'A' else np.sqrt(qm_res/(nA*nR))
        medias = df.groupby(fator)['Valor'].mean().sort_values(ascending=False)

        # ================= ANOVA DA REGRESSÃO =================
        if "regr_" in tipo_teste:
            if pd.to_numeric(df[fator], errors='coerce').isna().any(): 
                return {"status": "erro", "mensagem": "A regressão exige doses numéricas (quantitativas). Para tratamentos qualitativos, utilize os testes de médias."}
                
            rep_fator = nB * nR if fator == 'A' else nA * nR
            x = np.array(medias.index, dtype=float)
            y = np.array(medias.values, dtype=float)
            y_mean = np.mean(y)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, color='black', label='Médias Observadas', zorder=5)
            x_plot = np.linspace(min(x), max(x), 100)
            
            if modelo_regr == "linear":
                s, i, r, _, _ = stats.linregress(x, y)
                y_pred = i + s * x
                ax.plot(x_plot, i + s * x_plot, color='blue', label='Ajuste Linear', zorder=4)
                eq = f"y = {i:.4f} {'+' if s>=0 else '-'} {abs(s):.4f}x"
                r2 = r**2
                gl_reg = 1
            elif modelo_regr == "quadratica":
                c = np.polyfit(x, y, 2)
                p = np.poly1d(c)
                y_pred = p(x)
                ax.plot(x_plot, p(x_plot), color='red', label='Ajuste Quadrático', zorder=4)
                eq = f"y = {c[2]:.4f} {'+' if c[1]>=0 else '-'} {abs(c[1]):.4f}x {'+' if c[0]>=0 else '-'} {abs(c[0]):.4f}x²"
                ss_res_m = np.sum((y - y_pred)**2)
                ss_tot_m = np.sum((y - y_mean)**2)
                r2 = 1 - (ss_res_m / ss_tot_m)
                gl_reg = 2

            # Cálculos do Quadro de ANOVA específico para a Regressão
            sq_reg_raw = rep_fator * np.sum((y_pred - y_mean)**2)
            sq_trat_raw = sq_a if fator == 'A' else sq_b
            sq_desvio_raw = max(0, sq_trat_raw - sq_reg_raw)
            gl_trat = (nA - 1) if fator == 'A' else (nB - 1)
            gl_desvio = gl_trat - gl_reg

            qm_reg = sq_reg_raw / gl_reg
            qm_desvio = sq_desvio_raw / gl_desvio if gl_desvio > 0 else 0

            f_reg = qm_reg / qm_res if qm_res > 0 else 0
            f_desvio = qm_desvio / qm_res if qm_res > 0 else 0

            p_reg = 1 - stats.f.cdf(f_reg, gl_reg, gl_res)
            p_desvio = 1 - stats.f.cdf(f_desvio, gl_desvio, gl_res)

            anova_reg = [
                {"FV": f"Regressão ({modelo_regr.capitalize()})", "GL": gl_reg, "SQ": round(sq_reg_raw, 2), "QM": round(qm_reg, 2), "F Calc": round(f_reg, 2), "F Tab": calcular_f_tab(gl_reg, gl_res), "Sig": determinar_sig_texto(p_reg, gl_reg, gl_res)},
                {"FV": "Desvios da Regressão", "GL": gl_desvio, "SQ": round(sq_desvio_raw, 2), "QM": round(qm_desvio, 2), "F Calc": round(f_desvio, 2), "F Tab": calcular_f_tab(gl_desvio, gl_res), "Sig": determinar_sig_texto(p_desvio, gl_desvio, gl_res)},
                {"FV": "Resíduo (da ANOVA)", "GL": gl_res, "SQ": round(sq_res, 2), "QM": round(qm_res, 2), "F Calc": "-", "F Tab": "-", "Sig": "-"}
            ]

            ax.set_title(f"Ajuste {modelo_regr.capitalize()} - Fator {fator}")
            ax.set_xlabel("Doses")
            ax.set_ylabel("Produção / Resposta")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            b_png, b_pdf = io.BytesIO(), io.BytesIO()
            fig.savefig(b_png, format="png", bbox_inches='tight')
            fig.savefig(b_pdf, format="pdf", bbox_inches='tight')
            plt.close(fig)
            
            return {
                "status": "sucesso", 
                "tipo": "regressao", 
                "equacao": eq, 
                "r2": f"{r2:.4f}", 
                "modelo": modelo_regr.capitalize(), 
                "img_png": base64.b64encode(b_png.getvalue()).decode('utf-8'), 
                "img_pdf": base64.b64encode(b_pdf.getvalue()).decode('utf-8'),
                "anova_reg": anova_reg
            }

        if "tukey_" in tipo_teste:
            q_crit = qsturng(0.95, n_niveis, gl_res)
            dms = q_crit * ep
            return {"status": "sucesso", "tipo": "tukey", "q": round(q_crit, 4), "dms": round(dms, 4), "tabela": [{"Nível": str(n), "Média": round(v, 2)} for n, v in medias.items()]}
        
        if "duncan_" in tipo_teste:
            rp_duncan = {p: round(qsturng((0.95)**(p-1), p, gl_res) * ep, 4) for p in range(2, n_niveis + 1)}
            return {"status": "sucesso", "tipo": "duncan", "alcances": rp_duncan, "tabela": [{"Nível": str(n), "Média": round(v, 2)} for n, v in medias.items()]}

        if "dunnett_" in tipo_teste:
            if not testemunha or testemunha not in medias.index:
                return {"status": "erro", "mensagem": f"Testemunha '{testemunha}' não encontrada. Níveis válidos: {list(medias.index)}"}
            media_test = medias[testemunha]
            t_crit = stats.t.ppf(1 - (0.05 / (2 * (n_niveis - 1))), gl_res)
            dms_dunnett = t_crit * ep * np.sqrt(2)
            resultados = []
            for n, v in medias.items():
                if n == testemunha: continue
                dif = v - media_test
                sig = "Significativo (*)" if abs(dif) >= dms_dunnett else "ns"
                resultados.append({"Tratamento": str(n), "Média": round(v, 2), "Diferença": round(dif, 2), "Sig": sig})
            return {"status": "sucesso", "tipo": "dunnett", "dms": round(dms_dunnett, 4), "testemunha": testemunha, "media_testemunha": round(media_test, 2), "resultados": resultados}

    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

# ================= MÓDULO 3: PARCELAS SUBDIVIDIDAS =================
@app.post("/api/analise/parcelas")
async def analisar_parcelas(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try: 
            df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
        except: 
            df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
            
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
        sq_b = (nA * nR) * ((df.groupby('B')['Valor'].mean() - mg)**2).sum()
        sq_inter = nR * ((df.groupby(['A', 'B'])['Valor'].mean() - mg)**2).sum() - sq_a - sq_b
        sq_erro_b = max(0, sq_tot - sq_bloc - sq_a - sq_erro_a - sq_b - sq_inter)
        
        gl_a, gl_bloc, gl_erro_a = nA - 1, nR - 1, (nA - 1) * (nR - 1)
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
        
        anova = [
            {"FV": "Fator A (Parcela)", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(qm_a,2), "F Calc": round(f_a,2), "F Tab": calcular_f_tab(gl_a, gl_erro_a), "Sig": determinar_sig_texto(1-stats.f.cdf(f_a, gl_a, gl_erro_a), gl_a, gl_erro_a)},
            {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(qm_bloc,2), "F Calc": round(f_bloc,2), "F Tab": calcular_f_tab(gl_bloc, gl_erro_a), "Sig": determinar_sig_texto(1-stats.f.cdf(f_bloc, gl_bloc, gl_erro_a), gl_bloc, gl_erro_a)},
            {"FV": "Erro A (Parcela)", "GL": gl_erro_a, "SQ": round(sq_erro_a,2), "QM": round(qm_erro_a,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Fator B (Subparcela)", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(qm_b,2), "F Calc": round(f_b,2), "F Tab": calcular_f_tab(gl_b, gl_erro_b), "Sig": determinar_sig_texto(1-stats.f.cdf(f_b, gl_b, gl_erro_b), gl_b, gl_erro_b)},
            {"FV": "Interação AxB", "GL": gl_inter, "SQ": round(sq_inter,2), "QM": round(qm_inter,2), "F Calc": round(f_inter,2), "F Tab": calcular_f_tab(gl_inter, gl_erro_b), "Sig": determinar_sig_texto(1-stats.f.cdf(f_inter, gl_inter, gl_erro_b), gl_inter, gl_erro_b)},
            {"FV": "Erro B (Subparcela)", "GL": gl_erro_b, "SQ": round(sq_erro_b,2), "QM": round(qm_erro_b,2), "F Calc": "-", "F Tab": "-", "Sig": "-"},
            {"FV": "Total", "GL": gl_tot, "SQ": round(sq_tot,2), "QM": "-", "F Calc": "-", "F Tab": "-", "Sig": "-"}
        ]
        
        return {
            "status": "sucesso", 
            "cv_a": round((np.sqrt(qm_erro_a)/mg)*100,2), 
            "cv_b": round((np.sqrt(qm_erro_b)/mg)*100,2), 
            "anova": anova
        }
        
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
