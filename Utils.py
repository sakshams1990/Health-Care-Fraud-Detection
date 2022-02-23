def sort_shap_value(dic, reverse):
    return dict(sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse))
