import shap
import tensorflow as tf

# https://github.com/slundberg/shap/issues/2189#issuecomment-1048384801
tf.compat.v1.disable_v2_behavior()


def compute_shap(background, model, inputs):
    # e = shap.DeepExplainer(model, background)
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    e = shap.GradientExplainer(model, background)
    return e.shap_values(inputs)
