//! Device abstraction for TensorCrab.
//!
//! [`Device`] identifies where tensor data lives — either on the CPU (host
//! memory) or on a specific NVIDIA GPU.

/// The device where tensor data resides.
///
/// # Example
/// ```
/// use tensor_crab::device::Device;
///
/// let cpu = Device::Cpu;
/// assert_eq!(cpu.to_string(), "cpu");
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU host memory.
    #[default]
    Cpu,
    /// NVIDIA GPU at the given ordinal index.
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda(n) => write!(f, "cuda:{n}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn test_device_cpu_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_device_cuda_display() {
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(2).to_string(), "cuda:2");
    }

    #[test]
    fn test_device_clone_eq() {
        let a = Device::Cpu;
        let b = a.clone();
        assert_eq!(a, b);
    }
}
