from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from scipy import stats
import io

app = FastAPI(title="API Estatística - Fernando Paes Lorena")

# Configuração de CORS: Permite que o Frontend (Cloudflare) converse com este Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na versão final, você coloca a URL do seu site aqui
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def determinar_sig_texto(p_valor, gl_num, gl_den):
    if pd.isna(p_valor) or gl_den <= 0: return "ns (p >= 0,05)"
    if p_valor < 0.01: return "1% (**)"
    elif p_valor < 0.05: return "5% (*)"
    else: return "ns (p >= 0,05)"

@app.post("/api/analise/fatorial")
async def analisar_fatorial(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="O arquivo deve ser um CSV.")
    
    try:
        # Lê o arquivo que chegou pela internet
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), sep=';', decimal=',')
            if len(df.columns) == 1:
                df = pd.read_csv(io.BytesIO(content), sep=',', decimal='.')
        except:
            raise HTTPException(status_code=400, detail="Erro na leitura do CSV.")

        cols = list(df.columns)
        if len(cols) < 4:
            raise HTTPException(status_code=400, detail="CSV precisa de 4 colunas: FatorA, FatorB, Bloco, Valor.")

        df.rename(columns={cols[0]: 'A', cols[1]: 'B', cols[2]: 'Bloco', cols[3]: 'Valor'}, inplace=True)
        df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
        
        nA, nB, nR = len(df['A'].unique()), len(df['B'].unique()), len(df['Bloco'].unique())
        
        # --- CÁLCULOS DA ANOVA ---
        mg = df['Valor'].mean()
        sq_tot = ((df['Valor'] - mg)**2).sum()
        sq_bloc = (nA*nB) * ((df.groupby('Bloco')['Valor'].mean() - mg)**2).sum()
        sq_a = (nB*nR) * ((df.groupby('A')['Valor'].mean() - mg)**2).sum()
        sq_b = (nA*nR) * ((df.groupby('B')['Valor'].mean() - mg)**2).sum()
        sq_inter = nR * ((df.groupby(['A', 'B'])['Valor'].mean() - mg)**2).sum() - sq_a - sq_b
        sq_res = max(0, sq_tot - sq_bloc - sq_a - sq_b - sq_inter)
        
        gl_a, gl_b, gl_inter, gl_bloc, gl_tot = nA-1, nB-1, (nA-1)*(nB-1), nR-1, (nA*nB*nR)-1
        gl_res = gl_tot - gl_a - gl_b - gl_inter - gl_bloc
        
        qm_a, qm_b, qm_inter = sq_a/gl_a, sq_b/gl_b, sq_inter/gl_inter
        qm_res = sq_res/gl_res if gl_res > 0 else 0
        
        f_a = qm_a/qm_res if qm_res > 0 else 0
        f_b = qm_b/qm_res if qm_res > 0 else 0
        f_i = qm_inter/qm_res if qm_res > 0 else 0
        
        p_a = 1-stats.f.cdf(f_a, gl_a, gl_res)
        p_b = 1-stats.f.cdf(f_b, gl_b, gl_res)
        p_i = 1-stats.f.cdf(f_i, gl_inter, gl_res)

        # Montando o JSON de resposta
        anova_result = [
            {"FV": "Fator A", "GL": gl_a, "SQ": round(sq_a,2), "QM": round(qm_a,2), "F Calc": round(f_a,2), "Sig": determinar_sig_texto(p_a, gl_a, gl_res)},
            {"FV": "Fator B", "GL": gl_b, "SQ": round(sq_b,2), "QM": round(qm_b,2), "F Calc": round(f_b,2), "Sig": determinar_sig_texto(p_b, gl_b, gl_res)},
            {"FV": "Interação", "GL": gl_inter, "SQ": round(sq_inter,2), "QM": round(qm_inter,2), "F Calc": round(f_i,2), "Sig": determinar_sig_texto(p_i, gl_inter, gl_res)},
            {"FV": "Blocos", "GL": gl_bloc, "SQ": round(sq_bloc,2), "QM": round(sq_bloc/gl_bloc if gl_bloc>0 else 0,2), "F Calc": "-", "Sig": "-"},
            {"FV": "Resíduo", "GL": gl_res, "SQ": round(sq_res,2), "QM": round(qm_res,2), "F Calc": "-", "Sig": "-"}
        ]
        
        cv = ((np.sqrt(qm_res)/mg)*100)

        # A API devolve um dicionário que será convertido para JSON automaticamente
        return {
            "status": "sucesso",
            "cv": round(cv, 2),
            "configuracao": f"A({nA}), B({nB}), Blocos({nR})",
            "anova": anova_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))