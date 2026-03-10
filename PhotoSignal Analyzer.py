import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
import os
import signal
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.signal import find_peaks, peak_prominences
st.set_page_config(layout="wide")
st.title("🔬 Welcome to NOME DO PROGRAMA")
# Funções
# ------------------------------------------------------------



# -------------------------
# Funções utilitárias
# -------------------------
def make_odd(n):
    n = int(n)
    if n % 2 == 0:
        n += 1
    return max(3, n)

def safe_savgol(signal_arr, window_length, polyorder):
    N = len(signal_arr)
    if window_length >= N:
        window_length = make_odd(max(3, N - 1))
    if polyorder >= window_length:
        polyorder = max(1, window_length - 1)
    return savgol_filter(signal_arr, window_length, polyorder)

# -------------------------
# Caches para cálculos pesados
# -------------------------
@st.cache_data(show_spinner=False)
def calcular_psd_cached(signal_arr, fs, nperseg=1024):
    return welch(signal_arr, fs=fs, nperseg=nperseg)

@st.cache_data(show_spinner=False)
def rodar_irls_cached(grab_vals, iso_vals):
    X = sm.add_constant(iso_vals)
    model = sm.RLM(grab_vals, X, M=sm.robust.norms.TukeyBiweight(c=4))
    res = model.fit()
    return res

@st.cache_data(show_spinner=False)
def busca_savgol_cached(signal_arr, feature_samples, k_min, k_max, polyorder_base):
    candidates = []
    metrics = []
    N = len(signal_arr)
    # limitar k_max para evitar loop gigantesco
    k_max_safe = min(k_max, max(1, N // max(4, feature_samples)))
    for k in range(k_min, k_max_safe + 1):
        w = make_odd(k * feature_samples)
        if w >= N:
            continue
        filtered = safe_savgol(signal_arr, w, polyorder_base)
        resid = signal_arr - filtered
        resid_energy = float(np.sum(resid**2))
        second_deriv = np.diff(filtered, n=2)
        smoothness = float(np.sum(second_deriv**2)) if len(second_deriv) > 0 else 0.0
        candidates.append((int(w), int(polyorder_base)))
        metrics.append((resid_energy, smoothness))
    return candidates, metrics

@st.cache_data(show_spinner=False)
def lowpass_filter_cached(signal_arr, fs, f_cut, order=2):
    if f_cut <= 0 or f_cut >= fs/2:
        # invalid cutoff -> return original as safe fallback
        return signal_arr
    b, a = butter(N=order, Wn=f_cut/(fs/2), btype='low')
    return filtfilt(b, a, signal_arr)

# -------------------------
# Uploads e seleção de coluna
# -------------------------
st.subheader("1) Load Files.csv (GRAB e ISO)")
col1, col2 = st.columns(2)
with col1:
    grab_file = st.file_uploader("Load file GRAB.csv", type="csv")
with col2:
    iso_file = st.file_uploader("Load file ISO.csv", type="csv")

st.subheader("2) Select the Region of interest")
escolha = st.radio("Choose:", ("Region1G", "Region0R"))
coluna_selecionada = escolha

# -------------------------
# Pré-processamento leve (rápido) e checagens
# -------------------------
if grab_file is None or iso_file is None:
    st.info("Upload both files (GRAB and ISO) to proceed.")
    st.stop()

# Leitura (remover colunas opcionais com try/except para segurança)
grab = pd.read_csv(grab_file)
isos = pd.read_csv(iso_file)
for df in (grab, isos):
    for c in ['FrameCounter', 'LedState', 'Stimulation', 'Output0', 'Output1', 'Input0', 'Input1']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

# Garantir coluna Timestamp
if 'Timestamp' not in grab.columns or 'Timestamp' not in isos.columns:
    st.error("The files must contain a 'Timestamp' column.'.")
    st.stop()

grab['time(s)'] = grab['Timestamp'] - grab['Timestamp'].iloc[0]
isos['time(s)'] = isos['Timestamp'] - isos['Timestamp'].iloc[0]

# Selecionar colunas requeridas
if coluna_selecionada not in grab.columns or coluna_selecionada not in isos.columns:
    st.error(f"A coluna {coluna_selecionada} não existe em um dos arquivos.")
    st.stop()

# Mostrar cabeçalhos (pequeno preview)
st.write("Grab (preview):")
st.dataframe(grab[[ 'time(s)', coluna_selecionada ]].head())
st.write("ISO (preview):")
st.dataframe(isos[[ 'time(s)', coluna_selecionada ]].head())

# -------------------------
# Frequência de amostragem (rápido)
# -------------------------
dt_grab = np.mean(np.diff(grab['time(s)']))
dt_iso = np.mean(np.diff(isos['time(s)']))
fs1 = 1.0 / dt_grab if dt_grab > 0 else np.nan
fs2 = 1.0 / dt_iso if dt_iso > 0 else np.nan
fs = float(np.nanmean([fs1, fs2]))
st.metric("Average sampling frequency (Hz)", f"{fs:.3f}")

# -------------------------
# Cálculo de frequência de corte (cacheável)
# -------------------------
signal_vals = grab[coluna_selecionada].values
f, Pxx = calcular_psd_cached(signal_vals, fs)
cum_energy = np.cumsum(Pxx)
cum_energy /= cum_energy[-1]
f_cut_energy = float(f[np.where(cum_energy >= 0.95)[0][0]])

Pxx_smooth = gaussian_filter1d(Pxx, sigma=2)
Pxx_norm = (Pxx_smooth - np.min(Pxx_smooth)) / (np.max(Pxx_smooth) - np.min(Pxx_smooth))
mask = f <= 5
f_subset = f[mask]
Pxx_subset = Pxx_norm[mask]
x1, y1 = f_subset[0], Pxx_subset[0]
x2, y2 = f_subset[-1], Pxx_subset[-1]
distances = np.abs((y2 - y1)*f_subset - (x2 - x1)*Pxx_subset + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
f_cut_elbow = float(f_subset[np.argmax(distances)])
candidates_freq = np.array([f_cut_energy, f_cut_elbow])
candidates_freq = candidates_freq[(candidates_freq >= 0) & (candidates_freq <= fs/2)]
f_cut_default = float(np.median(candidates_freq)) if len(candidates_freq) > 0 else (f_cut_energy if not np.isnan(f_cut_energy) else 1.0)

# permitir user override
st.header("Cutoff frequency calculation (Hz)")
f_cut_user = st.number_input("Final cutoff frequency (Hz)", min_value=0.0, value=float(f_cut_default), format="%.3f")
f_cut_final = float(f_cut_user)

# mostrar gráfico PSD compacto
fig_psd, ax_psd = plt.subplots(figsize=(5,3))
ax_psd.plot(f, Pxx_norm, label='Normalized PSD (Sigla PSD)')
ax_psd.axvline(f_cut_energy, color='r', linestyle='--', label=f'Energia: {f_cut_energy:.2f} Hz')
ax_psd.axvline(f_cut_elbow, color='b', linestyle='--', label=f'Cotovelo: {f_cut_elbow:.2f} Hz')
ax_psd.axvline(f_cut_final, color='k', linestyle=':', label=f'Final: {f_cut_final:.2f} Hz')
ax_psd.legend(fontsize=8)
ax_psd.set_title('PSD (compact)')
st.pyplot(fig_psd)
plt.close(fig_psd)

# -------------------------
# Filtragem passa-baixa (cached)
# -------------------------
Grab_filtered = lowpass_filter_cached(grab[coluna_selecionada].values, fs, f_cut_final)
Isos_filtered = lowpass_filter_cached(isos[coluna_selecionada].values, fs, f_cut_final)

# truncar para mesmo tamanho
n = min(len(Grab_filtered), len(Isos_filtered))
Grab_filtered = pd.Series(Grab_filtered[:n])
Isos_filtered = pd.Series(Isos_filtered[:n])

# -------------------------
# Seção principal: pipeline caro executado apenas por botão


# ======================================================
# Inicialização do estado
# ======================================================
if "pipeline_ok" not in st.session_state:
    st.session_state.pipeline_ok = False

# ======================================================
# SEÇÃO PRINCIPAL – BOTÃO
# ======================================================
st.header("Pipeline robusto (clicar para executar)")

if st.button("▶️ Executar regressão IRLS, ΔF/F e detecção S-G"):
    with st.spinner("Executando... isto pode levar alguns segundos dependendo do tamanho dos dados"):
        try:
            # -----------------------------
            # IRLS
            # -----------------------------
            rlm_res = rodar_irls_cached(
                Grab_filtered.values,
                Isos_filtered.values
            )

            iso_fitt = pd.Series(rlm_res.fittedvalues)
            dFF = (Grab_filtered - iso_fitt) / (iso_fitt + 1e-12)
            Z_scor = (dFF - np.nanmean(dFF)) / (
                np.nanstd(dFF) if np.nanstd(dFF) != 0 else 1.0
            )

            t = grab["time(s)"][:len(dFF)].values
            signal_arr = dFF.values.copy()

            # -----------------------------
            # salvar no estado
            # -----------------------------
            st.session_state.iso_fitt = iso_fitt
            st.session_state.dFF = dFF
            st.session_state.Z_scor = Z_scor
            st.session_state.signal_arr = signal_arr
            st.session_state.t = t
            st.session_state.pipeline_ok = True

            st.success("IRLS e ΔF/F concluídos com sucesso.")

        except Exception as e:
            st.error(f"Erro durante o pipeline inicial: {e}")

# ======================================================
# CONTINUA APENAS SE PIPELINE OK
# ======================================================
if st.session_state.pipeline_ok:

    iso_fitt = st.session_state.iso_fitt
    dFF = st.session_state.dFF
    Z_scor = st.session_state.Z_scor
    signal_arr = st.session_state.signal_arr
    t = st.session_state.t
    N = len(signal_arr)

    # ==================================================
    # SIDEBAR – PARÂMETROS S-G
    # ==================================================
    st.sidebar.title("Parâmetros S-G (busca)")

    fs_sidebar = st.sidebar.number_input(
        "fs (Hz)", value=fs, format="%.3f"
    )
    feature_duration_s = st.sidebar.number_input(
        "Duração característica (s)", value=0.08, format="%.3f"
    )
    k_min = int(
        st.sidebar.number_input("k mínimo", value=2, step=1)
    )
    k_max_input = int(
        st.sidebar.number_input(
            "k máximo (limite automático aplicado)", value=20, step=1
        )
    )
    polyorder_base = int(
        st.sidebar.selectbox("polyorder padrão", [2, 3, 4], index=1)
    )

    feature_samples = max(
        1, int(round(feature_duration_s * fs_sidebar))
    )
    k_max = min(
        k_max_input,
        max(2, N // max(1, feature_samples)),
    )

    # ==================================================
    # BUSCA S-G (cacheada)
    # ==================================================
    candidates, metrics = busca_savgol_cached(
        signal_arr,
        feature_samples,
        k_min,
        k_max,
        polyorder_base,
    )

    if len(candidates) == 0:
        st.warning(
            "Nenhum candidato gerado — ajuste os parâmetros."
        )
        st.stop()

    rows = [
        {
            "window": w,
            "polyorder": p,
            "residual_energy": r,
            "smoothness": s,
        }
        for (w, p), (r, s) in zip(candidates, metrics)
    ]

    with st.expander("Mostrar tabela de candidatos (até 200 linhas)"):
        st.dataframe(pd.DataFrame(rows).head(200))

    idx_best_resid = int(
        np.argmin([m[0] for m in metrics])
    )

    # ==================================================
    # EXPLICAÇÕES
    # ==================================================
    st.subheader("ℹ️ Explicações rápidas")
    with st.sidebar.expander("How these S-G search parameters work"):
	    st.markdown(
		"""
		These parameters control how the Savitzky–Golay filter candidates are
		generated and evaluated during the automatic search.

		---
		**fs (Hz)**  
		Sampling frequency of the signal, in Hertz (samples per second).
		It is used to convert time-based quantities into a number of samples.

		---
		**Characteristic duration (s)**  
		Expected temporal duration of the signal features you want to preserve
		(for example, the typical width of a transient or event).
		
		This value is converted into a number of samples and sets the *scale*
		of the Savitzky–Golay windows explored during the search.

		👉 Practical rule: choose a duration close to the **typical event length**
		in your data.

		---
		**k minimum**  
		Lower bound on the window size multiplier.
		Prevents the search from testing windows that are too small and overly
		sensitive to noise.

		---
		**k maximum**  
		Upper bound on the window size multiplier.
		Larger values allow stronger smoothing but increase computation time and
		risk oversmoothing.

		To keep the search efficient and safe, the maximum value is automatically
		limited based on the signal length and the characteristic duration.

		---
		**Default polyorder**  
		Polynomial order used as a baseline during the search.
		Lower values emphasize smoothing; higher values preserve local curvature.

		---
		**Internal constraints (applied automatically)**  
		The algorithm converts the characteristic duration into a number of
		samples and ensures that:
		
		• window sizes remain valid and odd  
		• the search space does not exceed the signal length  
		• computation time remains reasonable  

		These safeguards prevent invalid or unstable filter configurations.
		"""
	    )


    with st.expander("What does 'Default = best candidate by residual_energy' mean?"):
	    st.markdown(
		"""
		By default, the algorithm automatically selects the parameter combination
		that **minimizes the residual energy** between the original signal and the
		filtered signal.

		In practice, multiple candidate filters are tested. For each candidate,
		the signal is filtered and the **residual** (original signal minus filtered
		signal) is computed. The residual energy—typically defined as the sum of
		squared residuals—quantifies how much of the original signal is *not*
		explained by the filtered version.

		The candidate with the **lowest residual energy** is chosen as the default
		because it provides a data-driven compromise between noise reduction and
		signal preservation.

		However, this automatic choice is intended as a **starting point**, not a
		final answer. A very low residual energy may still correspond to
		under-smoothing (retaining noise) or over-smoothing (removing meaningful
		dynamics). For this reason, visual inspection and domain knowledge are
		strongly recommended for the final adjustment.
		"""
	    )

    with st.expander("What is the Savitzky–Golay?"):
        st.markdown(
            """
        The Savitzky–Golay filter is a signal smoothing technique designed to reduce
        noise while preserving the **local shape** of the signal.

        Instead of averaging points (as in moving-average filters), it fits a
        low-degree polynomial to small, overlapping windows of the data and
        replaces each point with the value predicted by that polynomial. This
        approach allows the filter to preserve important features such as **peak
        height, width, position, and local dynamics**.

        Because of these properties, Savitzky–Golay filtering is widely used in
        experimental and biological data analysis, including neuroscience,
        photometry, EEG, fMRI, and spectroscopy, where maintaining the structure of
        transient events is critical.
        """
        )

    with st.expander("Window_length and polyorder (how to choose them properly)"):
	    st.markdown(
		"""
		**Window_length (window size)**  
		Defines how many data points are used at a time to locally fit the polynomial.
		
		• It must be an **odd number** (e.g., 5, 7, 11, 21).  
		• **Smaller windows** → follow the signal more closely, preserve fast events, but leave more noise.  
		• **Larger windows** → provide stronger smoothing, reduce noise, but may flatten peaks and blur transitions.
		
		👉 Practical rule: choose a window that roughly matches the **typical duration of the events** you want to preserve in the signal.

		---
		**Polyorder (polynomial order)**  
		Controls the complexity of the polynomial fitted within each window.
		
		• It must be **smaller than the window_length**.  
		• **Low orders (1–2)** → strong smoothing, captures global trends.  
		• **Medium orders (2–3)** → good balance between smoothing and shape preservation.  
		• **High orders (≥4)** → follow fine details, but may reintroduce noise.
		
		👉 Practical rule:  
		• Start with **polyorder = 2 or 3**  
		• Increase it only if real peaks or rapid changes are being visibly distorted.

		---
		**Quick summary**  
		• `window_length` controls **how much smoothing is applied over time**  
		• `polyorder` controls **how well local signal shape is preserved**

		A good choice reduces noise **without removing biologically meaningful events**.
		"""
	    )

    # ==================================================
    # AJUSTE MANUAL
    # ==================================================
    st.subheader("Escolha manual (ajuste final)")

    default_w = int(candidates[idx_best_resid][0])
    default_p = int(candidates[idx_best_resid][1])

    chosen_w = st.number_input(
        "Escolher window_length (ímpar):",
        value=default_w,
        step=2,
    )
    chosen_p = st.number_input(
        "Escolher polyorder:",
        value=default_p,
        min_value=1,
        max_value=int(chosen_w - 1),
    )

    baseline_final = safe_savgol(
        signal_arr, int(chosen_w), int(chosen_p)
    )
    corrected_final = signal_arr - baseline_final
    corrected_final = (corrected_final - np.mean(corrected_final)) / (
    np.std(corrected_final) if np.std(corrected_final) != 0 else 1
    )
    # ==================================================
    # FIGURA FINAL
    # ==================================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(t, Grab_filtered[: len(t)], label="GRAB filtrado")
    axes[0].plot(t, iso_fitt[: len(t)], label="ISO ajustado")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, dFF[: len(t)], label="ΔF/F original")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, Z_scor[: len(t)], label="Z-score")
    axes[2].legend(fontsize=8)

    axes[3].plot(
        t, corrected_final[: len(t)], label="Z-score (corrigido S-G)"
    )
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("t (s)")
    axes[0].set_ylabel("Amplitude")

    st.pyplot(fig)

    # ==================================================
    # DOWNLOAD
    # ==================================================
    nome_arquivo = st.text_input(
        "Nome do arquivo para salvar (sem extensão):",
        value="grafico",
    )

    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=300, bbox_inches="tight"
    )
    buffer.seek(0)

    st.download_button(
        "📥 Baixar figura",
        data=buffer,
        file_name=f"{nome_arquivo}.png",
        mime="image/png",
    )

    # salvar resultado final
    st.session_state.corrected_final = corrected_final
    st.success("Pipeline concluído. Resultado salvo na sessão.")


# -------------------------
# Operações pós-pipeline (sincronização, START, etc.)
# Apenas habilitar se o pipeline já foi executado e corrected_final disponível
# -------------------------
if 'corrected_final' in st.session_state:
    x_corrected = st.session_state['corrected_final']
    dff = pd.DataFrame({coluna_selecionada: x_corrected})
    dff['time'] = grab['time(s)'][:len(dff)].values

    st.header("Sincronização de CSV da Fotometria")
    arquivo_fm = st.file_uploader("Carregar CSV da fotometria", type=["csv"], key="fm_upload")
    if arquivo_fm is not None:
        time_foto = pd.read_csv(arquivo_fm, header=None).rename(columns={0: "h", 1: "min", 2: "s", 3: "ms"})
        time_foto["time(s)"] = time_foto["h"] * 3600 + time_foto["min"] * 60 + time_foto["s"] + time_foto["ms"] / 1000
        time_foto["delta_time(s)"] = time_foto["time(s)"] - time_foto["time(s)"].iloc[0]
        num_grab = len(dff)
        indices = np.linspace(0, len(time_foto) - 1, num=num_grab).astype(int)
        time_foto_corr = time_foto.iloc[indices].reset_index(drop=True)
        df_foto_corr = time_foto_corr.drop(columns=["time(s)"]).rename(columns={"delta_time(s)": "time(s)"})
        df_foto_corr[coluna_selecionada] = dff[coluna_selecionada].values
        mean_val, std_val = df_foto_corr[coluna_selecionada].mean(), df_foto_corr[coluna_selecionada].std()
        df_foto_corr[coluna_selecionada + "_zscore"] = ((df_foto_corr[coluna_selecionada] - mean_val) / std_val) if std_val != 0 else 0
        st.dataframe(df_foto_corr.head())
        st.download_button("Baixar CSV corrigido", df_foto_corr.to_csv(index=False).encode("utf-8"), "fotometria_corrigida.csv")
    
############################################ Escolha do arquivo Start  ####################################################    	
        # START optional
        st.subheader("START (opcional)")
        arquivo_start = st.file_uploader("Selecionar CSV start (opcional)", type=["csv"], key="start_upload")
############################################### Inicio da analise start se tiver ####################################################
#####################################################################################################################################
######################################################################################################################################
        if arquivo_start is not None:
            
            try:
           
                ttl_start = pd.read_csv(arquivo_start, header=None)

                # validação do arquivo
                if ttl_start.empty:
                    st.error("Arquivo START está vazio.")
                    st.stop()
                
                if ttl_start.shape[1] < 4:
                    st.error("Arquivo START precisa ter 4 colunas: h, min, s, ms.")
                    st.stop()
                
                ttl_start = ttl_start.rename(columns={0:"h",1:"min",2:"s",3:"ms"})
                ttl_start['time(s)'] = (
                    ttl_start['h'] * 3600 +
                    ttl_start['min'] * 60 +
                    ttl_start['s'] +
                    ttl_start['ms'] / 1000
                )
        
                st.success("Arquivo START carregado.")
        
                # sincronização simples: procurar índice com tolerância progressiva
                time_foto = time_foto_corr.copy()
                indice_TTS = None
        
                for e in np.arange(0, 1, 0.01):
                    for i in range(len(time_foto)):
                        if abs(time_foto['time(s)'].iloc[i] - ttl_start['time(s)'].iloc[0]) <= e:
                            indice_TTS = i
                            break
                    if indice_TTS is not None:
                        break
        
                if indice_TTS is not None:
                    df_tts = df_foto_corr.iloc[indice_TTS:].reset_index(drop=True)
                    st.dataframe(df_tts.head())
        
                    st.download_button(
                        "Baixar CSV pós-START",
                        df_tts.to_csv(index=False).encode("utf-8"),
                        "fotometria_pos_START.csv"
                    )
                    # ============================
                    # 🔧 Controles interativos
                    # ============================
                    dFF_trans = df_tts.copy()
                
                    st.subheader("🔧 Ajustes de visualização do gráfico")
                
                    # Valores padrão
                    x_min_default = float(dFF_trans['time(s)'].min())
                    x_max_default = float(dFF_trans['time(s)'].max())
                
                    y_min_default = float(dFF_trans['Region1G'].min())
                    y_max_default = float(dFF_trans['Region1G'].max())
                
                    # Inputs
                    x_min = st.number_input("xlim mínimo (t em segundos):", value=x_min_default)
                    x_max = st.number_input("xlim máximo (t em segundos):", value=x_max_default)
                
                    y_min = st.number_input("ylim mínimo (mV):", value=y_min_default)
                    y_max = st.number_input("ylim máximo (mV):", value=y_max_default)
                
                    # ============================
                    # 📊 Gerar o gráfico
                    # ============================
                
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                
                    ax.plot(dFF_trans['time(s)'], dFF_trans['Region1G'])
                    ax.set_xlabel("t (s)")
                    ax.set_ylabel("mV")
                
                    # Aplicar limites
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                
                    # Linha vertical TTS
                    tts_time = dFF_trans['time(s)'].iloc[indice_TTS]
                    ax.axvline(x=tts_time, color='black', linestyle='-')
                
                    ax.set_title("Sinal Region1G com marcação do instante TTS")
                
                    st.pyplot(fig)
                    # --- Configuração da Página ---
                    #st.set_page_config(
                        #page_title="Análise de Sinais",
                        #layout="wide"
                    #)
                    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
                    #### aqui começa depois de ajustar o df, ou seja começa a analise de picos########################
                    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
                    # --- Variáveis Iniciais ---
                    # A taxa de amostragem (fs) é definida aqui
                    fs = fs
                    df_resultados_manual = pd.DataFrame() # DataFrame global para armazenar resultados (Streamlit State)
                    
                    ## ⚙️ BLOCO: Seleção dos Parâmetros pelo Usuário (Sidebar)
                    # ==========================================================
                    
                    st.sidebar.header("Parâmetros de Detecção de Pico")
                    
                    params_default = {
                        "cutoff_z": 1.0,           # z-score cutoff para detectar picos
                        "prominence_thresh": 0.5,  # proeminência mínima
                        "janela_minimo": 50        # amostras ao redor do pico para mínimo
                    }
                            # expanders com explicações (não pesam)
                    st.subheader("ℹ️ Explicações rápidas")
                    with st.expander("O que significa cutoff_z?"):
                        st.markdown("""
                       # "Cutoff Z-score: valor mínimo do z-score para considerar um pico.").
                        """)
                    with st.expander("prominence_thresh?"):
                        st.markdown("Prominence Threshold: altura mínima relativa que um pico deve ter em relação à vizinhança.")
                    with st.expander("janela_minimo"):
                        st.markdown("Janela Mínima: número de amostras ao redor de cada pico para encontrar o mínimo local.")
                            
                    explicacoes = {
                        "cutoff_z": "Cutoff Z-score: valor mínimo do z-score para considerar um pico.",
                        "prominence_thresh": "Prominence Threshold: altura mínima relativa que um pico deve ter em relação à vizinhança.",
                        "janela_minimo": "Janela Mínima: número de amostras ao redor de cada pico para encontrar o mínimo local."
                    }
                    
                    # st.session_state é usado para manter o estado em Streamlit
                    if 'params' not in st.session_state:
                        st.session_state.params = params_default
                    
                    # --- Widgets para Entrada de Parâmetros ---
                    
                    st.sidebar.subheader("Z-Score Cutoff")
                    cutoff_z = st.sidebar.slider(
                        'Cutoff Z-score', 
                        min_value=0.1, 
                        max_value=5.0, 
                        value=st.session_state.params["cutoff_z"], 
                        step=0.1,
                        help=explicacoes["cutoff_z"]
                    )
                    
                    st.sidebar.subheader("Threshold de Proeminência")
                    prominence_thresh = st.sidebar.slider(
                        'Prominence Threshold', 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=st.session_state.params["prominence_thresh"], 
                        step=0.05,
                        help=explicacoes["prominence_thresh"]
                    )
                    
                    st.sidebar.subheader("Janela de Mínimo")
                    janela_minimo = st.sidebar.number_input(
                        'Janela Mínima (amostras)', 
                        min_value=1, 
                        max_value=500, 
                        value=st.session_state.params["janela_minimo"], 
                        step=10,
                        help=explicacoes["janela_minimo"]
                    )
                    
                    # Atualiza st.session_state
                    st.session_state.params.update({
                        "cutoff_z": cutoff_z,
                        "prominence_thresh": prominence_thresh,
                        "janela_minimo": int(janela_minimo)
                    })
                    
                    # --- Corpo Principal do App ---
                    
                    st.title("📊 Análise de Sinais")
                    st.write(f"**Taxa de Amostragem (fs):** {fs} Hz")
                    
                    st.subheader("Parâmetros Atuais")
                    st.json(st.session_state.params) # Mostra os parâmetros atuais
                    
                   
                    
                    ## 💻 Definição de Funções (Adaptação)
                    
                    # Função moving_average (mantida, mas com type hinting para clareza)
                    def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
                        """Calcula a média móvel de um sinal."""
                        if window_size <= 1:
                            return signal.copy()
                        # np.convolve é mantido, Streamlit é apenas a UI
                        return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
                    
                    
                    # Simulação do Controle do Programa (Toplevel/Botão de Fechar)
                    # Em Streamlit, o usuário simplesmente fecha a aba/browser, 
                    # mas um botão pode ser usado para reiniciar o estado.
                    
                    if st.sidebar.button("Limpar Parâmetros", help="Restaura os parâmetros para os valores default"):
                        st.session_state.params = params_default.copy()
                        st.experimental_rerun() # Força o rerun do app para aplicar os defaults
                    
                    #---
                    
                    ## ❓ Seleção do Modo de Análise (Substituindo `messagebox.askyesno`)
                    
                    st.header("Modo de Análise")
                    st.write("Deseja analisar a **SÉRIE COMPLETA** ou selecionar **JANELAS MANUALMENTE**?")
                    
                    col1, col2 = st.columns(2)
                    
                    # Simulação da caixa de diálogo "Sim/Não"
                    if col1.button("Sim, analisar SÉRIE COMPLETA", type="primary"):
                        st.session_state['escolha_analise'] = 'COMPLETA'
                        st.success("Modo de Análise: SÉRIE COMPLETA selecionado.")
                        
                    elif col2.button("Não, selecionar JANELAS MANUALMENTE"):
                        st.session_state['escolha_analise'] = 'MANUAL'
                        st.info("Modo de Análise: Seleção MANUAL de janelas selecionado.")
                    
                    # Lógica baseada na escolha (Aqui você colocaria a lógica de processamento)
                    if 'escolha_analise' in st.session_state:
                        st.subheader(f"Execução do Modo: {st.session_state['escolha_analise']}")
                        if st.session_state['escolha_analise'] == 'COMPLETA':
                            st.write("Executando análise para a série inteira com os parâmetros definidos...")
                            # Lógica para análise da série completa viria aqui
                        elif st.session_state['escolha_analise'] == 'MANUAL':
                            st.write("Pronto para o loop de seleção manual de janelas...")
                            # Lógica para o loop de seleção manual viria aqui
                            st.dataframe(df_resultados_manual) # Mostra o DF (vazio inicialmente)
                            
                        st.write("---")
                        st.write(f"**Parâmetros usados:** Cutoff Z={cutoff_z}, Prominence={prominence_thresh}, Janela Mínimo={janela_minimo}")
                        if 'escolha_analise' in st.session_state:
                            st.subheader(f"Execução do Modo: {st.session_state['escolha_analise']}")
#####################################################################################################################################
 #######################################################################################################################################
                        # Inicio da analise da serie completa
######################################################################################################################################
#####################################################################################################################################
                        if st.session_state['escolha_analise'] == 'COMPLETA':

                            st.write("Executando análise para a série inteira com os parâmetros definidos...")
                            df = dFF_trans
                        
                            colunas_z = [c for c in df.columns if c.endswith("_zscore")]
                            coluna = st.selectbox("Selecione o canal (coluna z-score):", colunas_z)
                        
                            y_original = df[coluna].values
                            tempos = df["time(s)"].values
                        
                            # zscore manual
                            mean_sig = np.mean(y_original)
                            std_sig = np.std(y_original)
                            
                            z_signal = (y_original - mean_sig) / std_sig if std_sig != 0 else y_original.copy()
                        
                            # Funções auxiliares
                            def lowpass_filter(signal, fs, f_cut):
                                from scipy.signal import butter, filtfilt
                                b, a = butter(4, f_cut / (fs / 2), btype='low')
                                return filtfilt(b, a, signal)
                        
                            def apply_moving_average_1d(signal, window_size=5):
                                kernel = np.ones(window_size) / window_size
                                return np.convolve(signal, kernel, mode='same')
                        
                            st.header("Análise de Frequência e Seleção de Filtro")
                        
                            col_analysis, col_options = st.columns([3, 1])
                        
                            # -------------------------------------------------------------
                            # BLOCO 1 — Cálculo de frequência de corte
                            # -------------------------------------------------------------
                            with col_analysis:
                        
                                st.subheader("Cálculo Automático da Frequência de Corte")
                        
                                f, Pxx = welch(z_signal, fs=fs, nperseg=1024)
                                cum_energy = np.cumsum(Pxx)
                                cum_energy /= cum_energy[-1]
                        
                                if np.any(cum_energy >= 0.95):
                                    f_cut_energy = f[np.where(cum_energy >= 0.95)[0][0]]
                                else:
                                    f_cut_energy = f[-1]
                        
                                Pxx_smooth = gaussian_filter1d(Pxx, sigma=2)
                        
                                f_cut_auto = f_cut_energy
                                st.info(f"Frequência de Corte Automática (95% da Energia): **{f_cut_auto:.2f} Hz**")
                        
                                f_cut_user = st.number_input("Frequência de corte final (Hz)",
                                                             min_value=0.0, value=float(f_cut_auto), format="%.3f")
                                f_cut_final = float(f_cut_user)
                        
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(f, Pxx, label='PSD Original')
                                ax.plot(f, Pxx_smooth, label='PSD Suavizada', linestyle='--', color='orange')
                                ax.axvline(f_cut_user, color='r', linestyle=':', label=f'Corte Automático {f_cut_user:.2f} Hz')
                                ax.set_xlim(0, 5)
                                ax.legend()
                                st.pyplot(fig)
                        
                            # -------------------------------------------------------------
                            # BLOCO 2 — Escolha do método de suavização
                            # -------------------------------------------------------------
                            with col_options:
                        
                                st.subheader("Escolha de Suavização")
                        
                                opcoes = {
                                    "Sim": "PASSA-BAIXA",
                                    "Não": "JANELA_MOVEL"
                                }
                        
                                escolha = st.radio("Qual método deseja usar?",
                                    options=list(opcoes.values()),
                                    index=0
                                )
                        
                                # ---------------------------------------
                                # ---------------------------------------
                                #  PASSA BAIXA
                                # ---------------------------------------
                                if escolha == "PASSA-BAIXA":
                                
                                    y = lowpass_filter(z_signal, fs, f_cut_final)
                                    y_ = lowpass_filter(y_original, fs, f_cut_final)
                                
                                    st.success("Método Selecionado: PASSA-BAIXA")
                                
                                    st.session_state['y_suavizado'] = y
                                    st.session_state['metodo_suavizacao'] = "PASSA-BAIXA"
                                
                                # ---------------------------------------
                                #  JANELA MÓVEL
                                # ---------------------------------------
                                elif escolha == "JANELA_MOVEL":
                                
                                    window_size = st.number_input(
                                        "Tamanho da janela da Média Móvel:",
                                        min_value=1,
                                        max_value=500,
                                        value=5,
                                        step=1,
                                        key="window_size_movel"
                                    )
                                
                                    y = apply_moving_average_1d(z_signal, int(window_size))
                                    y_ = apply_moving_average_1d(y_original, int(window_size))
                                
                                    st.success(f"Suavização concluída! Janela: {window_size}")
                                
                                    st.session_state['y_suavizado'] = y
                                    st.session_state['metodo_suavizacao'] = "JANELA_MOVEL"


                                # ===========================================================
                                # SEÇÃO FINAL — Mostrar o sinal suavizado + gráfico separado
                                # ===========================================================
                                
                            st.header("Resultado Final da Suavização")
                            
                            if 'y_suavizado' in st.session_state:
                            
                                y = st.session_state['y_suavizado']
                                metodo = st.session_state['metodo_suavizacao']
                            
                                st.subheader(f"Sinal Suavizado ({metodo})")
                                st.write("Valores do sinal tratados:")
                                st.write(y)
                            
                                x_min_default = float(df["time(s)"].min())
                                x_max_default = float(df["time(s)"].max())
                            
                                x_min = st.number_input("📌 Limite mínimo do eixo X",
                                                         value=x_min_default, key="final_xmin")
                                x_max = st.number_input("📌 Limite máximo do eixo X",
                                                         value=x_max_default, key="final_xmax")
                            
                                fig, ax = plt.subplots(figsize=(10, 3))
                               
                                ax.plot(df["time(s)"], y, label='Sinal Suavizado', color='red')
                                ax.set_xlim(x_min, x_max)
                                ax.legend()
                                st.pyplot(fig)
                                
                                st.header("Detectar picos")
                                # 4) Detectar picos e mínimos
                                picos, _ = find_peaks(y, height=cutoff_z,distance=2,width = 4)
                                minimos, _ = find_peaks(-y,distance=2, prominence=prominence_thresh,width = 4)
                                prominences = peak_prominences(y_, picos)[0]
                                picos_validos = picos[prominences > prominence_thresh]
                                picos_selecionados = picos_validos
                                st.write("Numero de picos",len(picos_selecionados))
                                # 5) Determinar linha de base para cada pico
                                indices_base_inicio = []
                                indices_base_fim = []
                                for pico in picos_selecionados:
                                        min_antes = minimos[minimos < pico]
                                        min_depois = minimos[minimos > pico]
                                    
                                        if len(min_antes) == 0 or len(min_depois) == 0:
                                            continue  # ignora picos sem base completa
                                    
                                        i0 = min_antes[-1]
                                        i1 = min_depois[0] if len(min_depois) else min(pico + int(0.1 * len(y)),len(y) - 1)

                                    
                                        indices_base_inicio.append(i0)
                                        indices_base_fim.append(i1)
                                # 6) Calcular área dos picos (acima da linha de base)
                                if len(picos) > 0:
                                    area_picos = sum([
                                        np.trapz(
                                            y[i0:i1+1] - (y[i0] + (y[i1]-y[i0])*(tempos[i0:i1+1]-tempos[i0])/(tempos[i1]-tempos[i0])),
                                            tempos[i0:i1+1]
                                        )
                                        for i0, i1 in zip(indices_base_inicio, indices_base_fim)
                                    ])
                                else:
                                    area_picos = 0.0
                        
                                # 7) Área total da janela
                                area_total = np.trapz(y, tempos)
                        
                                # 8) Área da base (região verde)
                                area_base = area_total - area_picos
                        
                                # 9) Média e desvio
                                media = np.mean(y)
                                desvio = np.std(y)
                        
                                
                                tempos = dFF_trans['time(s)'].values
                                x_min_default = float(tempos.min())
                                x_max_default = float(tempos.max())
                                
                                x_min = st.number_input(
                                    "📌 Limite mínimo do eixo X",
                                    value=x_min_default,
                                    key="final_xmin_picos"
                                )
                                
                                x_max = st.number_input(
                                    "📌 Limite máximo do eixo X",
                                    value=x_max_default,
                                    key="final_xmax_picos"
                                )
                                
                                
                                
                                # Sinal suavizado
                                ax.plot(dFF_trans['time(s)'], y, color="black", label="Sinal Suavizado")
                                
                                # Marcar os picos
                                ax.scatter(tempos[picos], y[picos], color="blue", zorder=5, label="Picos")
                                
                                ax.set_xlim(x_min, x_max)
                                
                                # Labels e título
                                ax.set_xlabel("Tempo (s)")
                                ax.set_ylabel("Z-score")
                                ax.set_title("Janela selecionada com suavização")
                                ax.legend()
                                
                                fig.tight_layout()
                                # --- nome escolhido pelo usuário ---
                                nome_arquivo = st.text_input(
                                    "Nome do arquivo para salvar (sem extensão):",
                                    value="grafico_picos"
                                )
                                nome_arquivo_final = f"{nome_arquivo}.png"
                                
                                # --- salvar figura na memória ---
                               
                                buffer = io.BytesIO()
                                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                                buffer.seek(0)
                                
                                # --- botão de download ---
                                st.download_button(
                                    label="📥 Baixar figura",
                                    data=buffer,
                                    file_name=nome_arquivo_final,
                                    mime="image/png"
                                )
                                
                                st.pyplot(fig)
                                # Cria figura para Streamlit
                                x_min = st.number_input(
                                    "📌 Limite mínimo do eixo X",
                                    value=x_min_default,
                                    key="final_xmin_picos_area"
                                )
                                
                                x_max = st.number_input(
                                    "📌 Limite máximo do eixo X",
                                    value=x_max_default,
                                    key="final_xmax_picos_area"
                                )
                                fig, ax = plt.subplots(figsize=(10, 5))
                               

                                

                                ax.plot(tempos, y, color="black", label="Suavizado")
                                area_total = 0
                                area_total = 0.0
                                n = min(
        		                    len(picos_selecionados),
        		                    len(indices_base_inicio),
        		                    len(indices_base_fim)
        		                )
                                for i in range(n):
                                    if not picos_selecionados[i]:
                                        continue
                                
                                    i0 = indices_base_inicio[i]
                                    i1 = indices_base_fim[i]
                                
                                    xi = tempos[i0:i1 + 1]
                                    yi = y[i0:i1 + 1]
                                
                                    # --- baseline linear entre os vales ---
                                    x0, y0 = tempos[i0], y[i0]
                                    x1, y1 = tempos[i1], y[i1]
                                
                                    if x1 != x0:
                                        m = (y1 - y0) / (x1 - x0)
                                    else:
                                        m = 0.0
                                
                                    base_line = y0 + m * (xi - x0)
                                
                                    # --- área do pico ---
                                    area = np.trapz(yi - base_line, xi)
                                    area_total += area
                                
                                    # --- preenchimento visual ---
                                    ax.fill_between(xi, yi, base_line, alpha=0.3)
                                

                                
                                # Configurações do gráfico
                                ax.set_xlabel("t(s)", fontsize=15)
                                ax.set_ylabel("Z-Score", fontsize=15)
                                ax.tick_params(axis='both', labelsize=15)
                                ax.set_xlim(x_min, x_max)
                                ax.set_title(f"Área Total dos Picos Selecionados: {area_total:.2f}")
                                ax.grid(True)
                                
                                # Exibir no Streamlit
                                 # --- nome escolhido pelo usuário ---
                                nome_arquivo = st.text_input(
                                    "Nome do arquivo para salvar (sem extensão):",
                                    value="grafico_picos_area"
                                )
                                nome_arquivo_final = f"{nome_arquivo}.png"
                                
                                # --- salvar figura na memória ---
                                
                                buffer = io.BytesIO()
                                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                                buffer.seek(0)
                                
                                # --- botão de download ---
                                st.download_button(
                                    label="📥 Baixar figura",
                                    data=buffer,
                                    file_name=nome_arquivo_final,
                                    mime="image/png"
                                )
                                st.pyplot(fig)
                                # 1) Encontrar picos e vales (mínimos)
                                picos1, _ = find_peaks(y)
                                vales1, _ = find_peaks(-y)
                                
                                # Garante que início e fim são vales
                                if 0 not in vales1:
                                    vales1 = np.insert(vales1, 0, 0)
                                if (len(y)-1) not in vales1:
                                    vales1 = np.append(vales1, len(y)-1)
                                
                                # 2) Pontos da linha de base (vales)
                                x_bases = tempos[vales1]
                                y_bases = y[vales1]
                                
                                # 3) Interpolação da linha de base
                                interp_base = interp1d(x_bases, y_bases, kind="linear", fill_value="extrapolate")
                                linha_base = interp_base(tempos)
                                
                                # 4) Área entre fundo e linha de base
                                fundo = np.min(y)
                                area_verde = np.trapz(linha_base - fundo, tempos)
                                
                                # ---------------------------------------------------------
                                # Cria figura para Streamlit
                                x_min = st.number_input(
                                    "📌 Limite mínimo do eixo X",
                                    value=x_min_default,
                                    key="final_xmin_picos_area_tonica"
                                )
                                
                                x_max = st.number_input(
                                    "📌 Limite máximo do eixo X",
                                    value=x_max_default,
                                    key="final_xmax_picos_area_tonica"
                                )
                                # Criar figura
                                fig, ax = plt.subplots(figsize=(10, 5))
                                
                                ax.plot(tempos, y, color='black', label='Sinal suavizado')
                                ax.plot(tempos, linha_base, color='limegreen', linewidth=2, label='Linha de base contínua')
                                ax.axhline(fundo, color='magenta', linestyle='--', label=f'Fundo (min z): {fundo:.2f}')
                                ax.fill_between(tempos, fundo, linha_base, color='limegreen', alpha=0.6, label='Área verde (total)')
                                
                                ax.set_title(f"Área total entre fundo ({fundo:.2f}) e linha de base: {area_verde:.2f}")
                                ax.set_xlabel("t(s)")
                                ax.set_ylabel("z-score")
                                ax.set_xlim(x_min, x_max)
                                ax.grid(True)
                                ax.legend()
                                
                                # Mostrar no Streamlit
                                st.pyplot(fig)
                                
                                # ---------------------------------------------------------
                                # NOME DO ARQUIVO ESCOLHIDO PELO USUÁRIO
                                nome_fig = st.text_input(
                                    "Nome do arquivo para salvar (sem extensão):",
                                    value="area_verde"
                                )
                                nome_final = f"{nome_fig}.png"
                                
                                # Gerar arquivo na memória
                                buffer = io.BytesIO()
                                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                                buffer.seek(0)
                                
                                # Botão para baixar
                                st.download_button(
                                    label="📥 Baixar figura (PNG)",
                                    data=buffer,
                                    file_name=nome_final,
                                    mime="image/png"
                                )
                                # --- Mostrar métricas ---
                                st.subheader("📊 Métricas – Série Completa")
                                
                                st.markdown(f"""
                                **Pontos analisados:** {len(y)}  
                                **Nº de picos:** {len(picos)}  
                                **Área dos picos:** {area_picos:.4f}  
                                **Área da base (região verde):** {area_verde:.4f}
                                """)
                                
                                # --- Opção de salvar y ---
                                salvar_y = st.checkbox("Salvar série suavizada (y) em CSV?")
                                
                                if salvar_y:
                                    st.warning("⚠️ Nenhum sinal suavizado disponível ainda.")
                                with st.expander("Finalize the program if you want to start analyzing by windows!"):
                                    st.markdown(
                                            """  """)
		                           
            
                                st.divider()
                                st.header("Encerrar Aplicativo (opcional)")
                                if st.button("Fechar programa"):
                                    st.success("Encerrando...")
                                    os.kill(os.getpid(), signal.SIGTERM)   
#########################################################################################################################################
#########################################################################################################################################
                        #FIM DA ANALISE DA SERIE COMPLETA
##########################################################################################################################################
########################################################################################################################################


#########################################################################################################################################
#########################################################################################################################################
                        #INICIO DA ANALISE POR JANELAS
##########################################################################################################################################
########################################################################################################################################
                                
                        st.header("🎚️ Configuração do Método de Suavização")
                        # =========================================================
                        # MÉTODO DE SUAVIZAÇÃO (INPUT DO USUÁRIO)
                        # =========================================================
                        
                        
                        metodo_input = st.radio(
                            "Escolha o método:",
                            ["JANELA_MOVEL", "PASSA-BAIXA"],
                            index=0,
                            key="metodo_suavizacao_input"   # 🔒 chave EXCLUSIVA de widget
                        )
                        
                        if metodo_input == "JANELA_MOVEL":
                            window_size = st.number_input(
                                "Tamanho da janela da média móvel:",
                                min_value=1, max_value=500, value=5, step=1,
                                key="window_movel_input"
                            )
                            st.session_state["params_suav"] = {
                                "metodo": "JANELA_MOVEL",
                                "window": int(window_size)
                            }
                        
                        elif metodo_input == "PASSA-BAIXA":
                            st.session_state["params_suav"] = {
                                "metodo": "PASSA-BAIXA"
                            }
                        
                        # =========================================================
                        # JANELA MANUAL
                        # =========================================================
                        st.subheader("🕒 Definição da Janela Temporal")
                        
                        tempo_usuario = st.number_input(
                            "Tempo central (s):",
                            min_value=float(df["time(s)"].min()),
                            max_value=float(df["time(s)"].max()),
                            value=float(df["time(s)"].mean()),
                            step=0.1,
                            key="tempo_central_input"
                        )
                        
                        janela_anterior = st.number_input(
                            "Janela ANTES (s):",
                            min_value=0.0, value=1.0, step=0.1,
                            key="janela_antes_input"
                        )
                        
                        janela_posterior = st.number_input(
                            "Janela DEPOIS (s):",
                            min_value=0.0, value=1.0, step=0.1,
                            key="janela_depois_input"
                        )
                        
                        # =========================================================
                        # APLICAR JANELA + SUAVIZAÇÃO
                        # =========================================================
                        if st.button("Aplicar janela e suavização", key="aplicar_btn"):
                        
                            t_min = tempo_usuario - janela_anterior
                            t_max = tempo_usuario + janela_posterior
                            mask = (df["time(s)"] >= t_min) & (df["time(s)"] <= t_max)
                            df_janela = df[mask]
                        
                            y_original = df_janela[coluna_selecionada].values
                        
                            z_signal = (y_original - np.mean(y_original)) / (
                                np.std(y_original) if np.std(y_original) != 0 else 1
                            )
                        
                            # Funções auxiliares
                            def lowpass_filter(signal, fs, f_cut):
                                from scipy.signal import butter, filtfilt
                                b, a = butter(4, f_cut / (fs / 2), btype="low")
                                return filtfilt(b, a, signal)
                        
                            def moving_average(signal, w):
                                kernel = np.ones(w) / w
                                return np.convolve(signal, kernel, mode="same")
                        
                            params = st.session_state["params_suav"]
                        
                            if params["metodo"] == "JANELA_MOVEL":
                                w = params["window"]
                                y = moving_average(z_signal, w)
                                metodo_final = f"JANELA MÓVEL (w={w})"
                        
                            elif params["metodo"] == "PASSA-BAIXA":
                                y = lowpass_filter(z_signal, fs, f_cut_final)
                                metodo_final = f"PASSA-BAIXA (fc={f_cut_final:.2f} Hz)"
                        
                            # Salvar resultados (NÃO usar chaves de widget)
                            st.session_state["y_suavizado"] = y
                            st.session_state["metodo_suavizacao_final"] = metodo_final
                            st.session_state["df_janela"] = df_janela
                        
                            st.success(f"Janela aplicada de {t_min:.2f}s a {t_max:.2f}s")
                            st.success(f"Método aplicado: {metodo_final}")
                        
                        # =========================================================
                        # RESULTADO FINAL + DETECÇÃO DE PICOS
                        # =========================================================
                        st.header("Resultado Final da Suavização")
                        # =========================================================
                        # RESULTADO FINAL DA SUAVIZAÇÃO
                        # =========================================================
                        if 'y_suavizado' in st.session_state:
                        
                            y = st.session_state['y_suavizado']
                            metodo = st.session_state.get(
                                'metodo_suavizacao_final',
                                st.session_state.get('metodo_suavizacao_input', 'Método não definido')
                            )
                        
                            st.subheader(f"Sinal Suavizado ({metodo})")
                            st.write("Valores do sinal tratados:")
                            st.write(y)
                        
                            tempos = df_janela['time(s)'].values
                        
                            x_min_default = float(tempos.min())
                            x_max_default = float(tempos.max())
                        
                            x_min = st.number_input("📌 Limite mínimo do eixo X",
                                                    value=x_min_default, key="final_xmin")
                            x_max = st.number_input("📌 Limite máximo do eixo X",
                                                    value=x_max_default, key="final_xmax")
                        
                            # -----------------------------------------------------
                            # Gráfico do sinal suavizado
                            # -----------------------------------------------------
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(tempos, y, label='Sinal Suavizado', color='red')
                            ax.set_xlim(x_min, x_max)
                            ax.legend()
                            st.pyplot(fig)
                        
                            st.header("Detectar picos")
                        
                            # -----------------------------------------------------
                            # 1) Detectar picos e mínimos
                            # -----------------------------------------------------
                            picos, _ = find_peaks(y, height=cutoff_z, distance=2, width=2)
                            minimos, _ = find_peaks(-y, distance=2, prominence=prominence_thresh, width=2)
                        
                            prominences = peak_prominences(y, picos)[0]
                            picos_selecionados = picos[prominences > prominence_thresh]
                        
                            st.write("Número de picos:", len(picos_selecionados))
                        
                            # -----------------------------------------------------
                            # 2) Determinar linha de base de cada pico
                            # -----------------------------------------------------
                             # 5) Determinar linha de base para cada pico
                            indices_base_inicio = []
                            indices_base_fim = []
                            for pico in picos_selecionados:
                                min_antes = minimos[minimos < pico]
                                min_depois = minimos[minimos > pico]
                            
                                if len(min_antes) == 0 or len(min_depois) == 0:
                                    continue  # ignora picos sem base completa
                            
                                i0 = min_antes[-1]
                                i1 = min_depois[0] if len(min_depois) else min(pico + int(0.1 * len(y)),len(y) - 1)
        
                            
                                indices_base_inicio.append(i0)
                                indices_base_fim.append(i1)
                            # 6) Calcular área dos picos (acima da linha de base)
                            if len(picos) > 0:
                                area_picos = sum([
                                    np.trapz(
                                        y[i0:i1+1] - (y[i0] + (y[i1]-y[i0])*(tempos[i0:i1+1]-tempos[i0])/(tempos[i1]-tempos[i0])),
                                        tempos[i0:i1+1]
                                    )
                                    for i0, i1 in zip(indices_base_inicio, indices_base_fim)
                                ])
                            else:
                                area_picos = 0.0
                            # 7) Área total da janela
                            area_total = np.trapz(y, tempos)
                    
                            # 8) Área da base (região verde)
                            area_base = area_total - area_picos
                    
                            # 9) Média e desvio
                            media = np.mean(y)
                            desvio = np.std(y)
                    
                            
                            tempos = df_janela['time(s)'].values
                            x_min_default = float(tempos.min())
                            x_max_default = float(tempos.max())
                            
                            x_min = st.number_input(
                                "📌 Limite mínimo do eixo X",
                                value=x_min_default,
                                key="final_xmin_picos"
                            )
                            
                            x_max = st.number_input(
                                "📌 Limite máximo do eixo X",
                                value=x_max_default,
                                key="final_xmax_picos"
                            )
                            
                            
                            
                            # Sinal suavizado
                            ax.plot(df_janela['time(s)'], y, color="black", label="Sinal Suavizado")
                            
                            # Marcar os picos
                            ax.scatter(tempos[picos], y[picos], color="blue", zorder=5, label="Picos")
                            
                            ax.set_xlim(x_min, x_max)
                            
                            # Labels e título
                            ax.set_xlabel("Tempo (s)")
                            ax.set_ylabel("Z-score")
                            ax.set_title("Janela selecionada com suavização")
                            ax.legend()
                            
                            fig.tight_layout()
                            # --- nome escolhido pelo usuário ---
                            nome_arquivo = st.text_input(
                                "Nome do arquivo para salvar (sem extensão):",
                                value="grafico_picos"
                            )
                            nome_arquivo_final = f"{nome_arquivo}.png"
                            
                            # --- salvar figura na memória ---
                           
                            buffer = io.BytesIO()
                            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                            buffer.seek(0)
                            
                            # --- botão de download ---
                            st.download_button(
                                label="📥 Baixar figura",
                                data=buffer,
                                file_name=nome_arquivo_final,
                                mime="image/png"
                            )
                            
                            st.pyplot(fig)
                            # Cria figura para Streamlit
                            x_min = st.number_input(
                                "📌 Limite mínimo do eixo X",
                                value=x_min_default,
                                key="final_xmin_picos_area"
                            )
                            
                            x_max = st.number_input(
                                "📌 Limite máximo do eixo X",
                                value=x_max_default,
                                key="final_xmax_picos_area"
                            )
                            fig, ax = plt.subplots(figsize=(10, 5))
                           
        
                            
        
                            ax.plot(tempos, y, color="black", label="Suavizado")
                            area_total = 0
                        
                            for i, sel in enumerate(picos_selecionados):
                                if sel:
                                    i0 = indices_base_inicio[i]
                                    i1 = indices_base_fim[i]
                            
                                    xi = tempos[i0:i1+1]
                                    yi = y[i0:i1+1]
                            
                                    # --- reta da base ---
                                    x0, y0 = tempos[i0], y[i0]
                                    x1, y1 = tempos[i1], y[i1]
                            
                                    m = (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else 0
                                    base_line = y0 + m * (xi - x0)
                            
                                    # --- área do pico ---
                                    area = np.trapz(yi - base_line, xi)
                                    area_total += area
                            
                                    # Preenchimento visual do pico
                                    ax.fill_between(xi, yi, base_line, alpha=0.3)
                            
                            # Configurações do gráfico
                            ax.set_xlabel("t(s)", fontsize=15)
                            ax.set_ylabel("Z-Score", fontsize=15)
                            ax.tick_params(axis='both', labelsize=15)
                            ax.set_xlim(x_min, x_max)
                            ax.set_title(f"Área Total dos Picos Selecionados: {area_total:.2f}")
                            ax.grid(True)
                            
                            # Exibir no Streamlit
                             # --- nome escolhido pelo usuário ---
                            nome_arquivo = st.text_input(
                                "Nome do arquivo para salvar (sem extensão):",
                                value="grafico_picos_area"
                            )
                            nome_arquivo_final = f"{nome_arquivo}.png"
                            
                            # --- salvar figura na memória ---
                            
                            buffer = io.BytesIO()
                            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                            buffer.seek(0)
                            
                            # --- botão de download ---
                            st.download_button(
                                label="📥 Baixar figura",
                                data=buffer,
                                file_name=nome_arquivo_final,
                                mime="image/png"
                            )
                            st.pyplot(fig)
                             # 1) Encontrar picos e vales (mínimos)
                            picos1, _ = find_peaks(y)
                            vales1, _ = find_peaks(-y)
                            
                            # Garante que início e fim são vales
                            if 0 not in vales1:
                                vales1 = np.insert(vales1, 0, 0)
                            if (len(y)-1) not in vales1:
                                vales1 = np.append(vales1, len(y)-1)
                            
                            # 2) Pontos da linha de base (vales)
                            x_bases = tempos[vales1]
                            y_bases = y[vales1]
                            
                            # 3) Interpolação da linha de base
                            interp_base = interp1d(x_bases, y_bases, kind="linear", fill_value="extrapolate")
                            linha_base = interp_base(tempos)
                            
                            # 4) Área entre fundo e linha de base
                            fundo = np.min(y)
                            area_verde = np.trapz(linha_base - fundo, tempos)
                            
                            # ---------------------------------------------------------
                            # Cria figura para Streamlit
                            x_min = st.number_input(
                                "📌 Limite mínimo do eixo X",
                                value=x_min_default,
                                key="final_xmin_picos_area_tonica"
                            )
                            
                            x_max = st.number_input(
                                "📌 Limite máximo do eixo X",
                                value=x_max_default,
                                key="final_xmax_picos_area_tonica"
                            )
                            # Criar figura
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            ax.plot(tempos, y, color='black', label='Sinal suavizado')
                            ax.plot(tempos, linha_base, color='limegreen', linewidth=2, label='Linha de base contínua')
                            ax.axhline(fundo, color='magenta', linestyle='--', label=f'Fundo (min z): {fundo:.2f}')
                            ax.fill_between(tempos, fundo, linha_base, color='limegreen', alpha=0.6, label='Área verde (total)')
                            
                            ax.set_title(f"Área total entre fundo ({fundo:.2f}) e linha de base: {area_verde:.2f}")
                            ax.set_xlabel("t(s)")
                            ax.set_ylabel("z-score")
                            ax.set_xlim(x_min, x_max)
                            ax.grid(True)
                            ax.legend()
                            
                            # Mostrar no Streamlit
                            st.pyplot(fig)
                            
                            # ---------------------------------------------------------
                            # NOME DO ARQUIVO ESCOLHIDO PELO USUÁRIO
                            nome_fig = st.text_input(
                                "Nome do arquivo para salvar (sem extensão):",
                                value="area_verde"
                            )
                            nome_final = f"{nome_fig}.png"
                            
                            # Gerar arquivo na memória
                            buffer = io.BytesIO()
                            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                            buffer.seek(0)
                            
                            # Botão para baixar
                            st.download_button(
                                label="📥 Baixar figura (PNG)",
                                data=buffer,
                                file_name=nome_final,
                                mime="image/png"
                            )
                            # --- Mostrar métricas ---
                            st.subheader("📊 Métricas – Série Completa")
                            
                            st.markdown(f"""
                            **Pontos analisados:** {len(y)}  
                            **Nº de picos:** {len(picos)}  
                            **Área dos picos:** {area_picos:.4f}  
                            **Área da base (região verde):** {area_verde:.4f}
                            """)
                            
                            # --- Opção de salvar y ---
                            salvar_y = st.checkbox("Salvar série suavizada (y) em CSV?")
                            
                            if salvar_y:
                                # Criar dataframe
                                df_y = pd.DataFrame({"time(s)": tempos, "y_suavizado": y})
                            
                                # Converter para CSV em memória
                                csv_bytes = df_y.to_csv(index=False).encode("utf-8")
                            
                                st.download_button(
                                    label="📥 Baixar y_suavizado.csv",
                                    data=csv_bytes,
                                    file_name="y_suavizado.csv",
                                    mime="text/csv"
                                )
                            
                            # --- Mostrar log como no Tkinter ---
                            st.subheader("📝 Log – Série Completa")
                            
                            resultado = f"""
                            Pontos analisados: {len(y)}
                            Nº de picos: {len(picos)}
                            Área dos picos: {area_picos:.4f}
                            Área da base (região verde): {area_verde:.4f}
                            """
                            
                            st.text(resultado)
                                                        
                                                                       
                               

                                            
                                   
 ############################################### Fim da escolha janela ######################################################################
#########################################################################################################################################                          
        				    
                    #analisar_tempo_manual(df_tts)
                else:
                    st.warning("Nenhum ponto correspondente ao START encontrado.")
                
            except Exception as e:
                st.error(f"Erro ao processar START {e}")
    
        
############################################### Fim do start se tiver ###################################################################
#########################################################################################################################################
#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                                               # inicio da analise sem o arquivo start 
########################################################################################################################################
        else:
            st.info("ℹ️ Nenhum arquivo START selecionado. Continuando sem ele.")
            df_tts = df_foto_corr
            st.write("✅ Continua sem ttl start", df_foto_corr.head())
            dFF_trans = df_tts
    # Valores padrão
            x_min_default = float(dFF_trans['time(s)'].min())
            x_max_default = float(dFF_trans['time(s)'].max())
        
            y_min_default = float(dFF_trans['Region1G'].min())
            y_max_default = float(dFF_trans['Region1G'].max())
        
            # Inputs
            x_min = st.number_input("xlim mínimo (t em segundos):", value=x_min_default)
            x_max = st.number_input("xlim máximo (t em segundos):", value=x_max_default)
        
            y_min = st.number_input("ylim mínimo (mV):", value=y_min_default)
            y_max = st.number_input("ylim máximo (mV):", value=y_max_default)
        
            # ============================
            # 📊 Gerar o gráfico
            # ============================
        
            fig, ax = plt.subplots(figsize=(12, 8))
            
        
            ax.plot(dFF_trans['time(s)'], dFF_trans['Region1G'])
            ax.set_xlabel("t (s)")
            ax.set_ylabel("mV")
        
            # Aplicar limites
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
            # Linha vertical TTS
            #tts_time = dFF_trans['time(s)'].iloc[indice_TTS]
            #ax.axvline(x=tts_time, color='black', linestyle='-')
        
            #ax.set_title("Sinal Region1G com marcação do instante TTS")
        
            st.pyplot(fig)
            # --- Configuração da Página ---
            #st.subheader("Análise de Sinais")
                
            #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
            #### aqui começa depois de ajustar o df, ou seja começa a analise de picos########################
            #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
            # --- Variáveis Iniciais ---
            # A taxa de amostragem (fs) é definida aqui
            fs = fs
            df_resultados_manual = pd.DataFrame() # DataFrame global para armazenar resultados (Streamlit State)
            
            ## ⚙️ BLOCO: Seleção dos Parâmetros pelo Usuário (Sidebar)
            # ==========================================================
            
            st.sidebar.header("Parâmetros de Detecção de Pico")
            
            params_default = {
                "cutoff_z": 1.0,           # z-score cutoff para detectar picos
                "prominence_thresh": 0.5,  # proeminência mínima
                "janela_minimo": 50        # amostras ao redor do pico para mínimo
            }
                    # expanders com explicações (não pesam)
            st.subheader("ℹ️ Explicações rápidas")
            with st.expander("O que significa cutoff_z?"):
                st.markdown("""
               # "Cutoff Z-score: valor mínimo do z-score para considerar um pico.").
                """)
            with st.expander("prominence_thresh?"):
                st.markdown("Prominence Threshold: altura mínima relativa que um pico deve ter em relação à vizinhança.")
            with st.expander("janela_minimo"):
                st.markdown("Janela Mínima: número de amostras ao redor de cada pico para encontrar o mínimo local.")
                    
            explicacoes = {
                "cutoff_z": "Cutoff Z-score: valor mínimo do z-score para considerar um pico.",
                "prominence_thresh": "Prominence Threshold: altura mínima relativa que um pico deve ter em relação à vizinhança.",
                "janela_minimo": "Janela Mínima: número de amostras ao redor de cada pico para encontrar o mínimo local."
            }
            
            # st.session_state é usado para manter o estado em Streamlit
            if 'params' not in st.session_state:
                st.session_state.params = params_default
            
            # --- Widgets para Entrada de Parâmetros ---
            
            st.sidebar.subheader("Z-Score Cutoff")
            cutoff_z = st.sidebar.slider(
                'Cutoff Z-score', 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.params["cutoff_z"], 
                step=0.1,
                help=explicacoes["cutoff_z"]
            )
            
            st.sidebar.subheader("Threshold de Proeminência")
            prominence_thresh = st.sidebar.slider(
                'Prominence Threshold', 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.params["prominence_thresh"], 
                step=0.05,
                help=explicacoes["prominence_thresh"]
            )
            
            st.sidebar.subheader("Janela de Mínimo")
            janela_minimo = st.sidebar.number_input(
                'Janela Mínima (amostras)', 
                min_value=1, 
                max_value=500, 
                value=st.session_state.params["janela_minimo"], 
                step=10,
                help=explicacoes["janela_minimo"]
            )
            
            # Atualiza st.session_state
            st.session_state.params.update({
                "cutoff_z": cutoff_z,
                "prominence_thresh": prominence_thresh,
                "janela_minimo": int(janela_minimo)
            })
            
            # --- Corpo Principal do App ---
            
            st.title("📊 Análise de Sinais")
            st.write(f"**Taxa de Amostragem (fs):** {fs} Hz")
            
            st.subheader("Parâmetros Atuais")
            st.json(st.session_state.params) # Mostra os parâmetros atuais
            
           
            
            ## 💻 Definição de Funções (Adaptação)
            
            # Função moving_average (mantida, mas com type hinting para clareza)
            def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
                """Calcula a média móvel de um sinal."""
                if window_size <= 1:
                    return signal.copy()
                # np.convolve é mantido, Streamlit é apenas a UI
                return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
            
            
            # Simulação do Controle do Programa (Toplevel/Botão de Fechar)
            # Em Streamlit, o usuário simplesmente fecha a aba/browser, 
            # mas um botão pode ser usado para reiniciar o estado.
            
            if st.sidebar.button("Limpar Parâmetros", help="Restaura os parâmetros para os valores default"):
                st.session_state.params = params_default.copy()
                st.experimental_rerun() # Força o rerun do app para aplicar os defaults
            
            #---
            
            ## ❓ Seleção do Modo de Análise (Substituindo `messagebox.askyesno`)
            
            st.header("Modo de Análise")
            st.write("Deseja analisar a **SÉRIE COMPLETA** ou selecionar **JANELAS MANUALMENTE**?")
            
            col1, col2 = st.columns(2)
            
            # Simulação da caixa de diálogo "Sim/Não"
            if col1.button("Sim, analisar SÉRIE COMPLETA", type="primary"):
                st.session_state['escolha_analise'] = 'COMPLETA'
                st.success("Modo de Análise: SÉRIE COMPLETA selecionado.")
                
            elif col2.button("Não, selecionar JANELAS MANUALMENTE"):
                st.session_state['escolha_analise'] = 'MANUAL'
                st.info("Modo de Análise: Seleção MANUAL de janelas selecionado.")
            
            # Lógica baseada na escolha (Aqui você colocaria a lógica de processamento)
            if 'escolha_analise' in st.session_state:
                st.subheader(f"Execução do Modo: {st.session_state['escolha_analise']}")
                if st.session_state['escolha_analise'] == 'COMPLETA':
                    st.write("Executando análise para a série inteira com os parâmetros definidos...")
                    # Lógica para análise da série completa viria aqui
                elif st.session_state['escolha_analise'] == 'MANUAL':
                    st.write("Pronto para o loop de seleção manual de janelas...")
                    # Lógica para o loop de seleção manual viria aqui
                    st.dataframe(df_resultados_manual) # Mostra o DF (vazio inicialmente)
                    
                st.write("---")
                st.write(f"**Parâmetros usados:** Cutoff Z={cutoff_z}, Prominence={prominence_thresh}, Janela Mínimo={janela_minimo}")
                if 'escolha_analise' in st.session_state:
                    st.subheader(f"Execução do Modo: {st.session_state['escolha_analise']}")
#####################################################################################################################################
#######################################################################################################################################
                # Inicio da analise da serie completa
######################################################################################################################################
#####################################################################################################################################
                if st.session_state['escolha_analise'] == 'COMPLETA':
                
                    st.write("Executando análise para a série inteira com os parâmetros definidos...")
                    df = dFF_trans
                
                    colunas_z = [c for c in df.columns if c.endswith("_zscore")]
                    coluna = st.selectbox("Selecione o canal (coluna z-score):", colunas_z)
                
                    y_original = df[coluna].values
                    tempos = df["time(s)"].values
                
                    # zscore manual
                    mean_sig = np.mean(y_original)
                    std_sig = np.std(y_original)
                    
                    z_signal = (y_original - mean_sig) / std_sig if std_sig != 0 else y_original.copy()
                
                    # Funções auxiliares
                    def lowpass_filter(signal, fs, f_cut):
                        from scipy.signal import butter, filtfilt
                        b, a = butter(4, f_cut / (fs / 2), btype='low')
                        return filtfilt(b, a, signal)
                
                    def apply_moving_average_1d(signal, window_size=5):
                        kernel = np.ones(window_size) / window_size
                        return np.convolve(signal, kernel, mode='same')
                
                    st.header("Análise de Frequência e Seleção de Filtro")
                
                    col_analysis, col_options = st.columns([3, 1])
                
                    # -------------------------------------------------------------
                    # BLOCO 1 — Cálculo de frequência de corte
                    # -------------------------------------------------------------
                    with col_analysis:
                
                        st.subheader("Cálculo Automático da Frequência de Corte")
                
                        f, Pxx = welch(z_signal, fs=fs, nperseg=1024)
                        cum_energy = np.cumsum(Pxx)
                        cum_energy /= cum_energy[-1]
                
                        if np.any(cum_energy >= 0.95):
                            f_cut_energy = f[np.where(cum_energy >= 0.95)[0][0]]
                        else:
                            f_cut_energy = f[-1]
                
                        Pxx_smooth = gaussian_filter1d(Pxx, sigma=2)
                
                        f_cut_auto = f_cut_energy
                        st.info(f"Frequência de Corte Automática (95% da Energia): **{f_cut_auto:.2f} Hz**")
                
                        f_cut_user = st.number_input("Frequência de corte final (Hz)",
                                                     min_value=0.0, value=float(f_cut_auto), format="%.3f")
                        f_cut_final = float(f_cut_user)
                
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(f, Pxx, label='PSD Original')
                        ax.plot(f, Pxx_smooth, label='PSD Suavizada', linestyle='--', color='orange')
                        ax.axvline(f_cut_user, color='r', linestyle=':', label=f'Corte Automático {f_cut_user:.2f} Hz')
                        ax.set_xlim(0, 5)
                        ax.legend()
                        st.pyplot(fig)
                
                    # -------------------------------------------------------------
                    # BLOCO 2 — Escolha do método de suavização
                    # -------------------------------------------------------------
                    with col_options:
                
                        st.subheader("Escolha de Suavização")
                
                        opcoes = {
                            "Sim": "PASSA-BAIXA",
                            "Não": "JANELA_MOVEL"
                        }
                
                        escolha = st.radio("Qual método deseja usar?",
                            options=list(opcoes.values()),
                            index=0
                        )
                
                        # ---------------------------------------
                        #  PASSA BAIXA
                        # ---------------------------------------
                        if escolha == "PASSA-BAIXA":
                
                            y = lowpass_filter(z_signal, fs, f_cut_final)
                            y_ = lowpass_filter(y_original,fs,f_cut_final) # sinal original sem z score
                            st.success("Método Selecionado: PASSA-BAIXA")
                
                            # Salva no session_state
                            st.session_state['y_suavizado'] = y
                            st.session_state['metodo_suavizacao'] = "PASSA-BAIXA"
                
                        # ---------------------------------------
                        #  JANELA MÓVEL
                        # ---------------------------------------
                        else:
                            window_size = st.number_input(
                                "Tamanho da janela da Média Móvel:",
                                min_value=1, max_value=500, value=5, step=1
                            )
                
                            y = apply_moving_average_1d(z_signal, int(window_size))
                            y_ = apply_moving_average_1d(y_original, int(window_size)) # sinal original sem z score
                            st.success(f"Suavização concluída! Janela: {window_size}")
                
                            # Salva no session_state
                            st.session_state['y_suavizado'] = y
                            st.session_state['metodo_suavizacao'] = "JANELA_MOVEL"

                        # ===========================================================
                        # SEÇÃO FINAL — Mostrar o sinal suavizado + gráfico separado
                        # ===========================================================
                        
                    st.header("Resultado Final da Suavização")
                    
                    if 'y_suavizado' in st.session_state:
                    
                        y = st.session_state['y_suavizado']
                        metodo = st.session_state['metodo_suavizacao']
                    
                        st.subheader(f"Sinal Suavizado ({metodo})")
                        st.write("Valores do sinal tratados:")
                        st.write(y)
                    
                        x_min_default = float(df["time(s)"].min())
                        x_max_default = float(df["time(s)"].max())
                    
                        x_min = st.number_input("📌 Limite mínimo do eixo X",
                                                 value=x_min_default, key="final_xmin")
                        x_max = st.number_input("📌 Limite máximo do eixo X",
                                                 value=x_max_default, key="final_xmax")
                    
                        fig, ax = plt.subplots(figsize=(10, 3))
                       
                        ax.plot(df["time(s)"], y, label='Sinal Suavizado', color='red')
                        ax.set_xlim(x_min, x_max)
                        ax.legend()
                        st.pyplot(fig)
                        
                        st.header("Detectar picos")
                        # 4) Detectar picos e mínimos
                        picos, _ = find_peaks(y, height=cutoff_z,distance=2,width = 4)
                        minimos, _ = find_peaks(-y,distance=2, prominence=prominence_thresh,width = 4)
                        prominences = peak_prominences(y_, picos)[0]
                        picos_validos = picos[prominences > prominence_thresh]
                        picos_selecionados = picos_validos
                        st.write("Numero de picos",len(picos_selecionados))
                        # 5) Determinar linha de base para cada pico
                        indices_base_inicio = []
                        indices_base_fim = []
                        for pico in picos_selecionados:
                                min_antes = minimos[minimos < pico]
                                min_depois = minimos[minimos > pico]
                            
                                if len(min_antes) == 0 or len(min_depois) == 0:
                                    continue  # ignora picos sem base completa
                            
                                i0 = min_antes[-1]
                                i1 = min_depois[0] if len(min_depois) else min(pico + int(0.1 * len(y)),len(y) - 1)

                            
                                indices_base_inicio.append(i0)
                                indices_base_fim.append(i1)
                        # 6) Calcular área dos picos (acima da linha de base)
                        if len(picos) > 0:
                            area_picos = sum([
                                np.trapz(
                                    y[i0:i1+1] - (y[i0] + (y[i1]-y[i0])*(tempos[i0:i1+1]-tempos[i0])/(tempos[i1]-tempos[i0])),
                                    tempos[i0:i1+1]
                                )
                                for i0, i1 in zip(indices_base_inicio, indices_base_fim)
                            ])
                        else:
                            area_picos = 0.0
                
                        # 7) Área total da janela
                        area_total = np.trapz(y, tempos)
                
                        # 8) Área da base (região verde)
                        area_base = area_total - area_picos
                
                        # 9) Média e desvio
                        media = np.mean(y)
                        desvio = np.std(y)
                
                        
                        tempos = dFF_trans['time(s)'].values
                        x_min_default = float(tempos.min())
                        x_max_default = float(tempos.max())
                        
                        x_min = st.number_input(
                            "📌 Limite mínimo do eixo X",
                            value=x_min_default,
                            key="final_xmin_picos"
                        )
                        
                        x_max = st.number_input(
                            "📌 Limite máximo do eixo X",
                            value=x_max_default,
                            key="final_xmax_picos"
                        )
                        
                        
                        
                        # Sinal suavizado
                        ax.plot(dFF_trans['time(s)'], y, color="black", label="Sinal Suavizado")
                        
                        # Marcar os picos
                        ax.scatter(tempos[picos], y[picos], color="blue", zorder=5, label="Picos")
                        
                        ax.set_xlim(x_min, x_max)
                        
                        # Labels e título
                        ax.set_xlabel("Tempo (s)")
                        ax.set_ylabel("Z-score")
                        ax.set_title("Janela selecionada com suavização")
                        ax.legend()
                        
                        fig.tight_layout()
                        # --- nome escolhido pelo usuário ---
                        nome_arquivo = st.text_input(
                            "Nome do arquivo para salvar (sem extensão):",
                            value="grafico_picos"
                        )
                        nome_arquivo_final = f"{nome_arquivo}.png"
                        
                        # --- salvar figura na memória ---
                       
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                        buffer.seek(0)
                        
                        # --- botão de download ---
                        st.download_button(
                            label="📥 Baixar figura",
                            data=buffer,
                            file_name=nome_arquivo_final,
                            mime="image/png"
                        )
                        
                        st.pyplot(fig)
                        # Cria figura para Streamlit
                        x_min = st.number_input(
                            "📌 Limite mínimo do eixo X",
                            value=x_min_default,
                            key="final_xmin_picos_area"
                        )
                        
                        x_max = st.number_input(
                            "📌 Limite máximo do eixo X",
                            value=x_max_default,
                            key="final_xmax_picos_area"
                        )
                        fig, ax = plt.subplots(figsize=(10, 5))
                       

                        

                        ax.plot(tempos, y, color="black", label="Suavizado")
                        area_total = 0
                        st.write(
				    "len(picos_selecionados):", len(picos_selecionados),
				    "len(indices_base_inicio):", len(indices_base_inicio),
				    "len(indices_base_fim):", len(indices_base_fim)
				)
			
                        area_total = 0.0

                        n = min(
                            len(picos_selecionados),
                            len(indices_base_inicio),
                            len(indices_base_fim)
                        )
                        
                        for i in range(n):
                            if not picos_selecionados[i]:
                                continue
                        
                            i0 = indices_base_inicio[i]
                            i1 = indices_base_fim[i]
                        
                            xi = tempos[i0:i1 + 1]
                            yi = y[i0:i1 + 1]
                        
                            # --- baseline linear entre os vales ---
                            x0, y0 = tempos[i0], y[i0]
                            x1, y1 = tempos[i1], y[i1]
                        
                            if x1 != x0:
                                m = (y1 - y0) / (x1 - x0)
                            else:
                                m = 0.0
                        
                            base_line = y0 + m * (xi - x0)
                        
                            # --- área do pico ---
                            area = np.trapz(yi - base_line, xi)
                            area_total += area
                        
                            # --- preenchimento visual ---
                            ax.fill_between(xi, yi, base_line, alpha=0.3)
                        
                        
                        # Configurações do gráfico
                        ax.set_xlabel("t(s)", fontsize=15)
                        ax.set_ylabel("Z-Score", fontsize=15)
                        ax.tick_params(axis='both', labelsize=15)
                        ax.set_xlim(x_min, x_max)
                        ax.set_title(f"Área Total dos Picos Selecionados: {area_total:.2f}")
                        ax.grid(True)
                        
                        # Exibir no Streamlit
                         # --- nome escolhido pelo usuário ---
                        nome_arquivo = st.text_input(
                            "Nome do arquivo para salvar (sem extensão):",
                            value="grafico_picos_area"
                        )
                        nome_arquivo_final = f"{nome_arquivo}.png"
                        
                        # --- salvar figura na memória ---
                        
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                        buffer.seek(0)
                        
                        # --- botão de download ---
                        st.download_button(
                            label="📥 Baixar figura",
                            data=buffer,
                            file_name=nome_arquivo_final,
                            mime="image/png"
                        )
                        st.pyplot(fig)
                        # 1) Encontrar picos e vales (mínimos)
                        picos1, _ = find_peaks(y)
                        vales1, _ = find_peaks(-y)
                        
                        # Garante que início e fim são vales
                        if 0 not in vales1:
                            vales1 = np.insert(vales1, 0, 0)
                        if (len(y)-1) not in vales1:
                            vales1 = np.append(vales1, len(y)-1)
                        
                        # 2) Pontos da linha de base (vales)
                        x_bases = tempos[vales1]
                        y_bases = y[vales1]
                        
                        # 3) Interpolação da linha de base
                        interp_base = interp1d(x_bases, y_bases, kind="linear", fill_value="extrapolate")
                        linha_base = interp_base(tempos)
                        
                        # 4) Área entre fundo e linha de base
                        fundo = np.min(y)
                        area_verde = np.trapz(linha_base - fundo, tempos)
                        
                        # ---------------------------------------------------------
                        # Cria figura para Streamlit
                        x_min = st.number_input(
                            "📌 Limite mínimo do eixo X",
                            value=x_min_default,
                            key="final_xmin_picos_area_tonica"
                        )
                        
                        x_max = st.number_input(
                            "📌 Limite máximo do eixo X",
                            value=x_max_default,
                            key="final_xmax_picos_area_tonica"
                        )
                        # Criar figura
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        ax.plot(tempos, y, color='black', label='Sinal suavizado')
                        ax.plot(tempos, linha_base, color='limegreen', linewidth=2, label='Linha de base contínua')
                        ax.axhline(fundo, color='magenta', linestyle='--', label=f'Fundo (min z): {fundo:.2f}')
                        ax.fill_between(tempos, fundo, linha_base, color='limegreen', alpha=0.6, label='Área verde (total)')
                        
                        ax.set_title(f"Área total entre fundo ({fundo:.2f}) e linha de base: {area_verde:.2f}")
                        ax.set_xlabel("t(s)")
                        ax.set_ylabel("z-score")
                        ax.set_xlim(x_min, x_max)
                        ax.grid(True)
                        ax.legend()
                        
                        # Mostrar no Streamlit
                        st.pyplot(fig)
                        
                        # ---------------------------------------------------------
                        # NOME DO ARQUIVO ESCOLHIDO PELO USUÁRIO
                        nome_fig = st.text_input(
                            "Nome do arquivo para salvar (sem extensão):",
                            value="area_verde"
                        )
                        nome_final = f"{nome_fig}.png"
                        
                        # Gerar arquivo na memória
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                        buffer.seek(0)
                        
                        # Botão para baixar
                        st.download_button(
                            label="📥 Baixar figura (PNG)",
                            data=buffer,
                            file_name=nome_final,
                            mime="image/png"
                        )
                        # --- Mostrar métricas ---
                        st.subheader("📊 Métricas – Série Completa")
                        
                        st.markdown(f"""
                        **Pontos analisados:** {len(y)}  
                        **Nº de picos:** {len(picos)}  
                        **Área dos picos:** {area_picos:.4f}  
                        **Área da base (região verde):** {area_verde:.4f}
                        """)
                        
                        # --- Opção de salvar y ---
                        salvar_y = st.checkbox("Salvar série suavizada (y) em CSV?")
                        
                        if salvar_y:
                            # Criar dataframe
                            df_y = pd.DataFrame({"time(s)": tempos, "y_suavizado": y})
                        
                            # Converter para CSV em memória
                            csv_bytes = df_y.to_csv(index=False).encode("utf-8")
                        
                            st.download_button(
                                label="📥 Baixar y_suavizado.csv",
                                data=csv_bytes,
                                file_name="y_suavizado.csv",
                                mime="text/csv"
                            )
                        
                        # --- Mostrar log como no Tkinter ---
                        st.subheader("📝 Log – Série Completa")
                        
                        resultado = f"""
                        Pontos analisados: {len(y)}
                        Nº de picos: {len(picos)}
                        Área dos picos: {area_picos:.4f}
                        Área da base (região verde): {area_verde:.4f}
                        """
                        
                        st.text(resultado)
                        






                        
                    else:
                        st.warning("⚠️ Nenhum sinal suavizado disponível ainda.")
                    with st.expander("Finalize the program if you want to start analyzing by windows!"):
                        st.markdown(
                            """
                        
                        """
                        )

                    st.divider()
                    st.header("Encerrar Aplicativo (opcional)")
                    if st.button("Fechar programa"):
                        st.success("Encerrando...")
                        os.kill(os.getpid(), signal.SIGTERM)
#########################################################################################################################################
#########################################################################################################################################
                #FIM DA ANALISE DA SERIE COMPLETA
##########################################################################################################################################
########################################################################################################################################


#########################################################################################################################################
#########################################################################################################################################
                #INICIO DA ANALISE POR JANELAS
##########################################################################################################################################
########################################################################################################################################
                        
                st.header("🎚️ Configuração do Método de Suavização")
                
                # =========================================================
                # MÉTODO DE SUAVIZAÇÃO (INPUT DO USUÁRIO)
                # =========================================================
                
                
                metodo_input = st.radio(
                    "Escolha o método:",
                    ["JANELA_MOVEL", "PASSA-BAIXA"],
                    index=0,
                    key="metodo_suavizacao_input"   # 🔒 chave EXCLUSIVA de widget
                )
                
                if metodo_input == "JANELA_MOVEL":
                    window_size = st.number_input(
                        "Tamanho da janela da média móvel:",
                        min_value=1, max_value=500, value=5, step=1,
                        key="window_movel_input"
                    )
                    st.session_state["params_suav"] = {
                        "metodo": "JANELA_MOVEL",
                        "window": int(window_size)
                    }
                
                elif metodo_input == "PASSA-BAIXA":
                    st.session_state["params_suav"] = {
                        "metodo": "PASSA-BAIXA"
                    }
                
                # =========================================================
                # JANELA MANUAL
                # =========================================================
                st.subheader("🕒 Definição da Janela Temporal")
                
                tempo_usuario = st.number_input(
                    "Tempo central (s):",
                    min_value=float(df["time(s)"].min()),
                    max_value=float(df["time(s)"].max()),
                    value=float(df["time(s)"].mean()),
                    step=0.1,
                    key="tempo_central_input"
                )
                
                janela_anterior = st.number_input(
                    "Janela ANTES (s):",
                    min_value=0.0, value=1.0, step=0.1,
                    key="janela_antes_input"
                )
                
                janela_posterior = st.number_input(
                    "Janela DEPOIS (s):",
                    min_value=0.0, value=1.0, step=0.1,
                    key="janela_depois_input"
                )
                
                # =========================================================
                # APLICAR JANELA + SUAVIZAÇÃO
                # =========================================================
                if st.button("Aplicar janela e suavização", key="aplicar_btn"):
                
                    t_min = tempo_usuario - janela_anterior
                    t_max = tempo_usuario + janela_posterior
                    mask = (df["time(s)"] >= t_min) & (df["time(s)"] <= t_max)
                    df_janela = df[mask]
                
                    y_original = df_janela[coluna_selecionada].values
                
                    z_signal = (y_original - np.mean(y_original)) / (
                        np.std(y_original) if np.std(y_original) != 0 else 1
                    )
                
                    # Funções auxiliares
                    def lowpass_filter(signal, fs, f_cut):
                        from scipy.signal import butter, filtfilt
                        b, a = butter(4, f_cut / (fs / 2), btype="low")
                        return filtfilt(b, a, signal)
                
                    def moving_average(signal, w):
                        kernel = np.ones(w) / w
                        return np.convolve(signal, kernel, mode="same")
                
                    params = st.session_state["params_suav"]
                
                    if params["metodo"] == "JANELA_MOVEL":
                        w = params["window"]
                        y = moving_average(z_signal, w)
                        metodo_final = f"JANELA MÓVEL (w={w})"
                
                    elif params["metodo"] == "PASSA-BAIXA":
                        y = lowpass_filter(z_signal, fs, f_cut_final)
                        metodo_final = f"PASSA-BAIXA (fc={f_cut_final:.2f} Hz)"
                
                    # Salvar resultados (NÃO usar chaves de widget)
                    st.session_state["y_suavizado"] = y
                    st.session_state["metodo_suavizacao_final"] = metodo_final
                    st.session_state["df_janela"] = df_janela
                
                    st.success(f"Janela aplicada de {t_min:.2f}s a {t_max:.2f}s")
                    st.success(f"Método aplicado: {metodo_final}")
                
                # =========================================================
                # RESULTADO FINAL + DETECÇÃO DE PICOS
                # =========================================================
                st.header("Resultado Final da Suavização")
                # =========================================================
                # RESULTADO FINAL DA SUAVIZAÇÃO
                # =========================================================
                if 'y_suavizado' in st.session_state:
                
                    y = st.session_state['y_suavizado']
                    metodo = st.session_state.get(
                        'metodo_suavizacao_final',
                        st.session_state.get('metodo_suavizacao_input', 'Método não definido')
                    )
                
                    st.subheader(f"Sinal Suavizado ({metodo})")
                    st.write("Valores do sinal tratados:")
                    st.write(y)
                
                    tempos = df_janela['time(s)'].values
                
                    x_min_default = float(tempos.min())
                    x_max_default = float(tempos.max())
                
                    x_min = st.number_input("📌 Limite mínimo do eixo X",
                                            value=x_min_default, key="final_xmin")
                    x_max = st.number_input("📌 Limite máximo do eixo X",
                                            value=x_max_default, key="final_xmax")
                
                    # -----------------------------------------------------
                    # Gráfico do sinal suavizado
                    # -----------------------------------------------------
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(tempos, y, label='Sinal Suavizado', color='red')
                    ax.set_xlim(x_min, x_max)
                    ax.legend()
                    st.pyplot(fig)
                
                    st.header("Detectar picos")
                
                    # -----------------------------------------------------
                    # 1) Detectar picos e mínimos
                    # -----------------------------------------------------
                    picos, _ = find_peaks(y, height=cutoff_z, distance=2, width=2)
                    minimos, _ = find_peaks(-y, distance=2, prominence=prominence_thresh, width=2)
                
                    prominences = peak_prominences(y, picos)[0]
                    picos_selecionados = picos[prominences > prominence_thresh]
                
                    st.write("Número de picos:", len(picos_selecionados))
                
                     # -----------------------------------------------------
                    # 2) Determinar linha de base de cada pico
                    # -----------------------------------------------------
                     # 5) Determinar linha de base para cada pico
                    indices_base_inicio = []
                    indices_base_fim = []
                    for pico in picos_selecionados:
                        min_antes = minimos[minimos < pico]
                        min_depois = minimos[minimos > pico]
                    
                        if len(min_antes) == 0 or len(min_depois) == 0:
                            continue  # ignora picos sem base completa
                    
                        i0 = min_antes[-1]
                        i1 = min_depois[0] if len(min_depois) else min(pico + int(0.1 * len(y)),len(y) - 1)

                    
                        indices_base_inicio.append(i0)
                        indices_base_fim.append(i1)
                    # 6) Calcular área dos picos (acima da linha de base)
                    if len(picos) > 0:
                        area_picos = sum([
                            np.trapz(
                                y[i0:i1+1] - (y[i0] + (y[i1]-y[i0])*(tempos[i0:i1+1]-tempos[i0])/(tempos[i1]-tempos[i0])),
                                tempos[i0:i1+1]
                            )
                            for i0, i1 in zip(indices_base_inicio, indices_base_fim)
                        ])
                    else:
                        area_picos = 0.0
                    # 7) Área total da janela
                    area_total = np.trapz(y, tempos)
            
                    # 8) Área da base (região verde)
                    area_base = area_total - area_picos
            
                    # 9) Média e desvio
                    media = np.mean(y)
                    desvio = np.std(y)
            
                    
                    tempos = df_janela['time(s)'].values
                    x_min_default = float(tempos.min())
                    x_max_default = float(tempos.max())
                    
                    x_min = st.number_input(
                        "📌 Limite mínimo do eixo X",
                        value=x_min_default,
                        key="final_xmin_picos"
                    )
                    
                    x_max = st.number_input(
                        "📌 Limite máximo do eixo X",
                        value=x_max_default,
                        key="final_xmax_picos"
                    )
                    
                    
                    
                    # Sinal suavizado
                    ax.plot(df_janela['time(s)'], y, color="black", label="Sinal Suavizado")
                    
                    # Marcar os picos
                    ax.scatter(tempos[picos], y[picos], color="blue", zorder=5, label="Picos")
                    
                    ax.set_xlim(x_min, x_max)
                    
                    # Labels e título
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Z-score")
                    ax.set_title("Janela selecionada com suavização")
                    ax.legend()
                    
                    fig.tight_layout()
                    # --- nome escolhido pelo usuário ---
                    nome_arquivo = st.text_input(
                        "Nome do arquivo para salvar (sem extensão):",
                        value="grafico_picos"
                    )
                    nome_arquivo_final = f"{nome_arquivo}.png"
                    
                    # --- salvar figura na memória ---
                   
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                    buffer.seek(0)
                    
                    # --- botão de download ---
                    st.download_button(
                        label="📥 Baixar figura",
                        data=buffer,
                        file_name=nome_arquivo_final,
                        mime="image/png"
                    )
                    
                    st.pyplot(fig)
                    # Cria figura para Streamlit
                    x_min = st.number_input(
                        "📌 Limite mínimo do eixo X",
                        value=x_min_default,
                        key="final_xmin_picos_area"
                    )
                    
                    x_max = st.number_input(
                        "📌 Limite máximo do eixo X",
                        value=x_max_default,
                        key="final_xmax_picos_area"
                    )
                    fig, ax = plt.subplots(figsize=(10, 5))
                   

                    

                    ax.plot(tempos, y, color="black", label="Suavizado")
                    area_total = 0
                
                    for i, sel in enumerate(picos_selecionados):
                        if sel:
                            i0 = indices_base_inicio[i]
                            i1 = indices_base_fim[i]
                    
                            xi = tempos[i0:i1+1]
                            yi = y[i0:i1+1]
                    
                            # --- reta da base ---
                            x0, y0 = tempos[i0], y[i0]
                            x1, y1 = tempos[i1], y[i1]
                    
                            m = (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else 0
                            base_line = y0 + m * (xi - x0)
                    
                            # --- área do pico ---
                            area = np.trapz(yi - base_line, xi)
                            area_total += area
                    
                            # Preenchimento visual do pico
                            ax.fill_between(xi, yi, base_line, alpha=0.3)
                    
                    # Configurações do gráfico
                    ax.set_xlabel("t(s)", fontsize=15)
                    ax.set_ylabel("Z-Score", fontsize=15)
                    ax.tick_params(axis='both', labelsize=15)
                    ax.set_xlim(x_min, x_max)
                    ax.set_title(f"Área Total dos Picos Selecionados: {area_total:.2f}")
                    ax.grid(True)
                    
                    # Exibir no Streamlit
                     # --- nome escolhido pelo usuário ---
                    nome_arquivo = st.text_input(
                        "Nome do arquivo para salvar (sem extensão):",
                        value="grafico_picos_area"
                    )
                    nome_arquivo_final = f"{nome_arquivo}.png"
                    
                    # --- salvar figura na memória ---
                    
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                    buffer.seek(0)
                    
                    # --- botão de download ---
                    st.download_button(
                        label="📥 Baixar figura",
                        data=buffer,
                        file_name=nome_arquivo_final,
                        mime="image/png"
                    )
                    st.pyplot(fig)
                     # 1) Encontrar picos e vales (mínimos)
                    picos1, _ = find_peaks(y)
                    vales1, _ = find_peaks(-y)
                    
                    # Garante que início e fim são vales
                    if 0 not in vales1:
                        vales1 = np.insert(vales1, 0, 0)
                    if (len(y)-1) not in vales1:
                        vales1 = np.append(vales1, len(y)-1)
                    
                    # 2) Pontos da linha de base (vales)
                    x_bases = tempos[vales1]
                    y_bases = y[vales1]
                    
                    # 3) Interpolação da linha de base
                    interp_base = interp1d(x_bases, y_bases, kind="linear", fill_value="extrapolate")
                    linha_base = interp_base(tempos)
                    
                    # 4) Área entre fundo e linha de base
                    fundo = np.min(y)
                    area_verde = np.trapz(linha_base - fundo, tempos)
                    
                    # ---------------------------------------------------------
                    # Cria figura para Streamlit
                    x_min = st.number_input(
                        "📌 Limite mínimo do eixo X",
                        value=x_min_default,
                        key="final_xmin_picos_area_tonica"
                    )
                    
                    x_max = st.number_input(
                        "📌 Limite máximo do eixo X",
                        value=x_max_default,
                        key="final_xmax_picos_area_tonica"
                    )
                    # Criar figura
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    ax.plot(tempos, y, color='black', label='Sinal suavizado')
                    ax.plot(tempos, linha_base, color='limegreen', linewidth=2, label='Linha de base contínua')
                    ax.axhline(fundo, color='magenta', linestyle='--', label=f'Fundo (min z): {fundo:.2f}')
                    ax.fill_between(tempos, fundo, linha_base, color='limegreen', alpha=0.6, label='Área verde (total)')
                    
                    ax.set_title(f"Área total entre fundo ({fundo:.2f}) e linha de base: {area_verde:.2f}")
                    ax.set_xlabel("t(s)")
                    ax.set_ylabel("z-score")
                    ax.set_xlim(x_min, x_max)
                    ax.grid(True)
                    ax.legend()
                    
                    # Mostrar no Streamlit
                    st.pyplot(fig)
                    
                    # ---------------------------------------------------------
                    # NOME DO ARQUIVO ESCOLHIDO PELO USUÁRIO
                    nome_fig = st.text_input(
                        "Nome do arquivo para salvar (sem extensão):",
                        value="area_verde"
                    )
                    nome_final = f"{nome_fig}.png"
                    
                    # Gerar arquivo na memória
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                    buffer.seek(0)
                    
                    # Botão para baixar
                    st.download_button(
                        label="📥 Baixar figura (PNG)",
                        data=buffer,
                        file_name=nome_final,
                        mime="image/png"
                    )
                    # --- Mostrar métricas ---
                    st.subheader("📊 Métricas – Série Completa")
                    
                    st.markdown(f"""
                    **Pontos analisados:** {len(y)}  
                    **Nº de picos:** {len(picos)}  
                    **Área dos picos:** {area_picos:.4f}  
                    **Área da base (região verde):** {area_verde:.4f}
                    """)
                    
                    # --- Opção de salvar y ---
                    salvar_y = st.checkbox("Salvar série suavizada (y) em CSV?")
                    
                    if salvar_y:
                        # Criar dataframe
                        df_y = pd.DataFrame({"time(s)": tempos, "y_suavizado": y})
                    
                        # Converter para CSV em memória
                        csv_bytes = df_y.to_csv(index=False).encode("utf-8")
                    
                        st.download_button(
                            label="📥 Baixar y_suavizado.csv",
                            data=csv_bytes,
                            file_name="y_suavizado.csv",
                            mime="text/csv"
                        )
                    
                    # --- Mostrar log como no Tkinter ---
                    st.subheader("📝 Log – Série Completa")
                    
                    resultado = f"""
                    Pontos analisados: {len(y)}
                    Nº de picos: {len(picos)}
                    Área dos picos: {area_picos:.4f}
                    Área da base (região verde): {area_verde:.4f}
                    """
                    
                    st.text(resultado)
        
                else:
                    st.info("Aplique a janela e a suavização para visualizar os resultados.")

                
                
                    





                            
        
############################################### Fim da escolha janela ######################################################################
#########################################################################################################################################   
    
            




# -------------------------
# Botão para encerrar (opcional)



        

           
                           
                                
                     

                                                                     
                    
                    





                            
        
############################################### Fim da escolha janela ######################################################################
#########################################################################################################################################   
    
            




# -------------------------
# Botão para encerrar (opcional)
# -------------------------
st.divider()
st.header("Encerrar Aplicativo (opcional)")
if st.button("Fechar programa"):
    st.success("Encerrando...")
    os.kill(os.getpid(), signal.SIGTERM)

