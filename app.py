import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from groq import Groq
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
import io

# Carrega as variáveis de ambiente (onde deve estar a sua GROQ_API_KEY)
load_dotenv()

# --- Função Auxiliar ---
def obter_configuracao(chave):
    if chave in st.secrets:
        return st.secrets[chave]
    return os.getenv(chave)

st.set_page_config(page_title="Gupy CV Optimizer Pro", page_icon="🎯", layout="wide")

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #C0392B 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(192,57,43,0.3);
    }
    .metric-card h2 { font-size: 2.5rem; margin: 0; font-weight: 700; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.9; }

    .checklist-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 0.95rem;
    }
    .check-ok  { background: #e8f5e9; color: #2e7d32; }
    .check-no  { background: #fff3e0; color: #e65100; }

    .teaser-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .teaser-box h3 { color: #FF6B6B; margin-bottom: 6px; }

    .keyword-pill {
        display: inline-block;
        background: #fde8e8;
        color: #C0392B;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #f8b4b4;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #C0392B;
        border-bottom: 2px solid #fde8e8;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Funções de Extração ---
def extrair_texto_pdf(ficheiro_pdf):
    leitor = PyPDF2.PdfReader(ficheiro_pdf)
    texto = ""
    for pagina in leitor.pages:
        texto += pagina.extract_text() or ""
    return texto

def contar_secoes(texto):
    secoes = {
        "Resumo / Sobre":  ["sobre", "about", "summary", "resumo"],
        "Experiência":     ["experiência", "experience", "cargo", "empresa"],
        "Formação":        ["formação", "education", "graduação", "universidade", "faculdade"],
        "Competências":    ["competências", "skills", "habilidades", "conhecimentos"],
        "Certificações":   ["certificação", "certification", "certificado", "licença"],
        "Idiomas":         ["idioma", "language", "língua", "inglês", "espanhol", "português"]
    }
    texto_lower = texto.lower()
    return {s: any(p in texto_lower for p in palavras) for s, palavras in secoes.items()}

def extrair_palavras_chave(texto, top_n=20):
    stopwords = set([
        "de","a","o","e","do","da","em","um","para","com","uma","os","no",
        "se","na","por","mais","as","dos","das","ao","aos","que","não","ou",
        "seu","sua","seus","suas","ele","ela","eles","elas","eu","nós","este",
        "esta","isso","aqui","ter","ser","foi","como","mas","pelo","pela",
        "entre","após","sobre","até","desde","quando","também","porque",
        "então","ainda","já","muito","bem","anos","ano","mesmo","todo",
        "toda","minha","meu","contact","page","linkedin","profile"
    ])
    palavras = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', texto.lower())
    filtradas = [p for p in palavras if p not in stopwords]
    return Counter(filtradas).most_common(top_n)

def gerar_nuvem_palavras(texto):
    # Reutilizando a lógica de stopwords da função acima
    wc = WordCloud(
        width=900, height=300,
        background_color="white",
        colormap="Reds",
        max_words=80,
        prefer_horizontal=0.8,
        collocations=False,
    )
    wc.generate(texto)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def otimizar_curriculo_gupy(texto_perfil, texto_vaga):
    # Prompt de Sistema rigorosamente baseado no Manual da Gupy
    prompt_sistema = """Você é um Especialista em Recrutamento e Algoritmos ATS, especializado na plataforma Gupy.
Seu objetivo é analisar o currículo do candidato e otimizá-lo para uma vaga específica, garantindo que ele seja bem ranqueado pelos Agentes de IA da Gupy.

REGRAS OBRIGATÓRIAS BASEADAS NO MANUAL DA GUPY:
1. FOCO NO ALGORITMO: Os campos de Experiência e Habilidades são os que têm peso na ordenação. Você deve garantir que as palavras-chave da descrição da vaga apareçam organicamente no resumo e nas experiências do candidato (Técnica do Espelho).
2. NOMENCLATURAS PADRÃO: Ajuste os títulos dos cargos para nomenclaturas padrão de mercado para facilitar a leitura da IA.
3. DESTAQUE DE RESULTADOS: Na descrição das experiências, reescreva os tópicos para destacar resultados alcançados, métricas e ferramentas utilizadas.
4. HONESTIDADE: Não invente habilidades ou experiências que não constam no perfil original do candidato. Apenas realce e reestruture o que já existe para dar "match" com a vaga.
5. CARTA DE APRESENTAÇÃO ("Apresente-se"): A Gupy permite uma mensagem personalizada. Crie um texto de apresentação curto, engajador e direto ao ponto, conectando o perfil do candidato com a dor/necessidade da empresa. Sem clichês (evite "sou proativo", "fora da caixa").

SUA RESPOSTA DEVE CONTER EXATAMENTE A SEGUINTE ESTRUTURA:
## 📊 Análise de Compatibilidade (Match)
[Dê uma nota de 0 a 100% de match real entre o currículo atual e a vaga e explique o porquê em 2 linhas]

## 🔑 Palavras-Chave Faltantes
[Liste as palavras-chave da vaga que o candidato DEVE adicionar ao seu perfil se tiver conhecimento nelas]

## 📝 Currículo Otimizado para Gupy (Copiar e Colar)
[Escreva a versão final do currículo, formatada de forma limpa, com Resumo, Experiências (com bullet points focados em resultados) e Habilidades]

## 🎯 Texto para a seção "Apresente-se" (Carta de Apresentação)
[O texto persuasivo e personalizado para o candidato usar na inscrição]"""

    try:
        api_key = obter_configuracao("GROQ_API_KEY")
        if not api_key:
            return "Erro: GROQ_API_KEY não encontrada no arquivo .env ou secrets."
            
        client = Groq(api_key=api_key)
        resposta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user",   "content": f"DESCRIÇÃO DA VAGA:\n{texto_vaga}\n\n---\n\nPERFIL DO CANDIDATO:\n{texto_perfil}"}
            ],
            temperature=0.3, # Temperatura mais baixa para garantir aderência estrita aos fatos do currículo
            max_tokens=3000,
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"Ocorreu um erro ao comunicar com a API do Groq: {e}"

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

st.title("🎯 Gupy CV Optimizer Pro")
st.markdown("**Bata o Algoritmo (Agente de IA) da Gupy!** Faça o upload do seu currículo/LinkedIn e cole a descrição da vaga para gerar um perfil otimizado baseado nas diretrizes oficiais da Gupy.")

# Layout de duas colunas para os inputs
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.markdown('<p class="section-header">1. Seu Currículo / Perfil</p>', unsafe_allow_html=True)
    ficheiro_upload = st.file_uploader("📎 Faça upload do PDF (Ex: Perfil do LinkedIn)", type=["pdf"])

with col_input2:
    st.markdown('<p class="section-header">2. A Vaga Alvo</p>', unsafe_allow_html=True)
    texto_vaga = st.text_area("📋 Cole aqui a descrição completa da vaga desejada:", height=150)

if ficheiro_upload is not None and texto_vaga.strip() != "":
    with st.spinner("📄 A ler e processar o documento..."):
        texto_extraido  = extrair_texto_pdf(ficheiro_upload)
        secoes          = contar_secoes(texto_extraido)
        top_palavras    = extrair_palavras_chave(texto_extraido, top_n=15)

    st.success("✅ Dados carregados com sucesso!")
    st.divider()

    # ── PRÉ-ANÁLISE RÁPIDA ──────────────────────────────────────────
    st.subheader("🔍 Status do Currículo Original")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<p class="section-header">✅ Seções Obrigatórias Gupy</p>', unsafe_allow_html=True)
        st.caption("A Gupy foca pesadamente em Experiência e Competências.")
        for secao, presente in secoes.items():
            icon = "✅" if presente else "⚠️"
            cls  = "check-ok" if presente else "check-no"
            msg  = "Detectado" if presente else "Faltando"
            st.markdown(f'<div class="checklist-item {cls}">{icon} <b>{secao}</b> — {msg}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">☁️ Termos de Maior Peso no seu PDF</p>', unsafe_allow_html=True)
        if len(texto_extraido.strip()) > 50:
            img_buf = gerar_nuvem_palavras(texto_extraido)
            st.image(img_buf, width="stretch")

    st.divider()

    # CTA para análise IA
    st.markdown("### 🚀 Gerar Currículo Otimizado para Gupy")
    st.markdown("A IA usará a **Técnica do Espelho** recomendada no manual da Gupy para cruzar as suas experiências com os requisitos da vaga, gerando um currículo focado em resultados e uma carta de apresentação estratégica.")

    if st.button("✨ Otimizar Perfil Agora (Llama 3.3)", type="primary", use_container_width=True):
        with st.spinner("🤖 Analisando requisitos e reescrevendo currículo... Isso pode levar alguns segundos."):
            resultado_analise = otimizar_curriculo_gupy(texto_extraido, texto_vaga)

        st.divider()
        st.header("🏆 Seu Novo Currículo Otimizado")
        st.markdown(resultado_analise)
        
elif ficheiro_upload is not None and texto_vaga.strip() == "":
    st.warning("⚠️ Cole a descrição da vaga no campo ao lado para prosseguir com a otimização da IA.")
