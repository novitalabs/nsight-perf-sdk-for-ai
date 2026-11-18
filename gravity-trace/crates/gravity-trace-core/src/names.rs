//! Kernel / activity name helpers: interning, demangling, display shortening.
use cpp_demangle::{DemangleOptions, Symbol};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Interns names so aggregation keys are small integers.
#[derive(Default)]
pub struct Interner {
    ids: FxHashMap<Arc<str>, u32>,
    names: Vec<Arc<str>>,
}

impl Interner {
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = u32::try_from(self.names.len()).expect("fewer than 2^32 distinct names");
        let arc: Arc<str> = Arc::from(name);
        self.names.push(arc.clone());
        self.ids.insert(arc, id);
        id
    }

    pub fn get(&self, id: u32) -> &Arc<str> {
        &self.names[id as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Demangle an Itanium C++ symbol (CUDA kernels); returns the input when it is not mangled.
pub fn demangle(name: &str) -> String {
    if !name.starts_with("_Z") {
        return name.to_string();
    }
    match Symbol::new(name) {
        Ok(sym) => sym.demangle_with_options(&DemangleOptions::default()).unwrap_or_else(|_| name.to_string()),
        Err(_) => name.to_string(),
    }
}

/// Short display name for metric labels: demangled, without the parameter list, bounded length.
pub fn short_name(name: &str, max_len: usize) -> String {
    let full = if name.starts_with("_Z") {
        match Symbol::new(name) {
            Ok(sym) => sym
                .demangle_with_options(&DemangleOptions::default().no_params())
                .unwrap_or_else(|_| name.to_string()),
            Err(_) => name.to_string(),
        }
    } else {
        name.to_string()
    };
    if full.chars().count() <= max_len {
        return full;
    }
    let mut s: String = full.chars().take(max_len.saturating_sub(1)).collect();
    s.push('…');
    s
}

/// Cache for repeated demangling during conversion.
#[derive(Default)]
pub struct DemangleCache {
    map: FxHashMap<String, Arc<str>>,
}

impl DemangleCache {
    pub fn get(&mut self, name: &str) -> Arc<str> {
        if let Some(v) = self.map.get(name) {
            return v.clone();
        }
        let v: Arc<str> = Arc::from(demangle(name));
        self.map.insert(name.to_string(), v.clone());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demangles_cuda_kernels() {
        assert_eq!(demangle("_Z4tinyPfi"), "tiny(float*, int)");
        assert_eq!(short_name("_Z4tinyPfi", 64), "tiny");
        assert_eq!(demangle("Memcpy HtoD"), "Memcpy HtoD");
    }

    #[test]
    fn interner_dedups() {
        let mut i = Interner::default();
        let a = i.intern("a");
        let b = i.intern("b");
        assert_eq!(i.intern("a"), a);
        assert_ne!(a, b);
        assert_eq!(&**i.get(b), "b");
    }
}
