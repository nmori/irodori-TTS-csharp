using System.Reflection;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.Utils;
using static TorchSharp.torch;

namespace IrodoriTts.Model;

/// <summary>
/// Where the model's own layers create their parameters. Inside <see cref="OnMeta"/> they are created on the
/// meta device: no memory and no random initialization, which the checkpoint would overwrite anyway (4 s and a
/// second copy of the weights for v4.1-Small). <see cref="Materialize"/> then gives each parameter real storage.
/// The setting is per thread (unlike <c>torch.set_default_device</c>), so other threads keep working normally.
/// </summary>
public static class ModelInit
{
    [ThreadStatic] private static Device? _device;

    private static readonly FieldInfo InternalParams = typeof(nn.Module).GetField("_internal_params", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingFieldException("TorchSharp nn.Module._internal_params not found.");

    /// <summary>Device for new parameters (null: the default, CPU).</summary>
    public static Device? Device => _device;

    /// <summary>Create parameters on the meta device until the result is disposed.</summary>
    public static IDisposable OnMeta() => new Scope(torch.META);

    public static Tensor Zeros(params long[] shape) => torch.zeros(shape, device: _device);

    public static Tensor Ones(params long[] shape) => torch.ones(shape, device: _device);

    private sealed class Scope : IDisposable
    {
        private readonly Device? _previous = _device;

        public Scope(Device device) => _device = device;

        public void Dispose() => _device = _previous;
    }

    /// <summary>
    /// Replace every meta parameter of <paramref name="root"/> with an uninitialized CPU tensor of the same shape
    /// (to be filled from the checkpoint), floating-point ones in <paramref name="floatingDType"/> when given.
    /// Throws if a meta tensor is left anywhere in the modules.
    /// </summary>
    public static void Materialize(nn.Module root, ScalarType? floatingDType = null)
    {
        foreach (var module in root.named_modules().Select(m => m.module).Prepend(root))
        {
            var dict = (OrderedDict<string, Parameter>)InternalParams.GetValue(module)!;
            foreach (var name in module.named_parameters(recurse: false).Select(p => p.name).ToList())
            {
                var old = dict[name];
                if (old.device_type != DeviceType.META) continue;
                var dtype = floatingDType is ScalarType f && torch.is_floating_point(old) ? f : old.dtype;
                var fresh = new Parameter(torch.empty(old.shape, dtype: dtype), requires_grad: old.requires_grad);
                if (!ReplaceField(module, old, fresh))
                {
                    throw new InvalidOperationException($"No field holds parameter '{name}' of {module.GetType().Name}.");
                }
                dict[name] = fresh;
                old.Dispose();
            }
            foreach (var field in Fields(module.GetType()))
            {
                if (field.GetValue(module) is Tensor t && !t.IsInvalid && t.device_type == DeviceType.META)
                {
                    throw new InvalidOperationException($"{module.GetType().Name}.{field.Name} is still on the meta device.");
                }
            }
        }
    }

    private static IEnumerable<FieldInfo> Fields(Type? type)
    {
        for (; type is not null && type != typeof(nn.Module); type = type.BaseType)
        {
            foreach (var f in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (typeof(Tensor).IsAssignableFrom(f.FieldType)) yield return f;
            }
        }
    }

    private static bool ReplaceField(nn.Module module, Parameter old, Parameter fresh)
    {
        bool any = false;
        foreach (var field in Fields(module.GetType()))
        {
            if (ReferenceEquals(field.GetValue(module), old) && field.FieldType.IsAssignableFrom(typeof(Parameter)))
            {
                field.SetValue(module, fresh);
                any = true;
            }
        }
        return any;
    }
}
