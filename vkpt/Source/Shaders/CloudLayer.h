// Copyright (c) 2026 QuakeRay contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// The cloud layer the sky is made of (rt_sky_clouds): its shape, and how far the
// sun reaches into it.
//
// The slab is held over the eye, at a height the host sets, but its clouds are
// anchored in the world's horizontal plane, so that a cloud hangs over the same
// spot of the world however the eye moves -- which is what lets the sky
// (CmSkyClouds.comp) and the pass that puts the layer's shadow on the world
// (CmCloudShadow.comp) march the very same clouds: the two work in the same space,
// with the eye only as the origin of the rays. A cloud is an fBm of value noise
// whose distribution is flattened out, so that a coverage of N leaves N of the sky
// clear. Everything here is a function of the position it is handed alone.
//
// The world is the Quake one -- up is z, and the horizontal plane is x and y --
// and a point of it is handed in with one difference: the height is measured from
// the eye rather than from the origin of the world, the layer being held over the
// eye. The horizontal coordinates are the world's own, which is what puts a cloud
// over a fixed spot of the world as the eye moves; the height, being the eye's,
// slides the layer through the noise as the eye climbs.
//
// Both note that the optical depth of a cloud towards the sun depends on where
// the ray to the sun crosses the base of the layer, not on the height it is asked
// from: a sun ray is a straight line, so the same column of cloud is what any two
// points on it see.
//

// The noise is scaled by the thickness of the layer: the size of a cloud is then
// a fraction of the layer's depth, and a taller layer is a deeper one rather than
// one whose clouds are bigger.
const float CLOUD_FREQUENCY = 2.0;
const int   CLOUD_OCTAVES = 3;
const int   CLOUD_DETAIL_OCTAVES = 2;
const float CLOUD_DETAIL_FREQUENCY = 4.5;

// How far the fBm of value noise spreads around its own mean. Flattening the
// distribution by it (see cloudShape below) is what makes the coverage threshold
// mean "this fraction of the sky is denser than it".
const float CLOUD_NOISE_MEAN = 0.5;
const float CLOUD_NOISE_SIGMA = 0.12;
// Gives the cloud a solid core and turns what is left of the edge into the thin
// part of it.
const float CLOUD_SHAPE_POWER = 1.6;

// Extinction of a cloud of density 1, in layer thicknesses: a cloud that fills a
// tenth of the layer then has an optical depth of 0.8.
const float CLOUD_EXTINCTION = 8.0;

// Opening angle of the cone the sun is marched through, in radians of offset per
// unit of distance: it is what makes a cloud's edge brighten gradually as the sun
// comes out from behind it, instead of the whole cloud switching at once.
const float CLOUD_LIGHT_CONE = 0.15;

// The steps of the sequences the taps of a march are spread by, taken from the
// golden ratio: successive taps land in different parts of the stretch they stand
// for (a low-discrepancy sequence), rather than anywhere at all (white noise),
// which leaves a smaller and a finer error behind.
const float CLOUD_SEQUENCE_STEP = 0.6180339887; // along the march

// Where a march to the sun samples the layer: the place it has walked to, offset
// inside the cone the sunlight of a cloud comes from -- what a cloud sees of the
// sun is the sky around the sun rather than the sun alone, and the cone is what
// keeps the edge of a cloud's shadow soft. The offset is a spiral, one turn per
// step: a fixed pattern of taps rather than a random one, so that the map of the
// layer's shadow (CmCloudShadow.comp) and the march the sky falls back on where
// that map does not reach read the same light out of the same column.
vec3 cloudSunSample(vec3 p, vec3 sunDir, float distance, float stepSize)
{
    vec3 helper = abs(sunDir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, sunDir));
    vec3 bitangent = cross(sunDir, tangent);

    float angle = distance / max(stepSize, 1.0e-3) * 2.39996323; // the golden angle
    float radius = CLOUD_LIGHT_CONE * distance * 0.5;

    return p + sunDir * distance + (cos(angle) * tangent + sin(angle) * bitangent) * radius;
}

// --- noise -------------------------------------------------------------------

float hash13(vec3 p)
{
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float valueNoise3(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1, 0, 0));
    float n010 = hash13(i + vec3(0, 1, 0));
    float n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1));
    float n101 = hash13(i + vec3(1, 0, 1));
    float n011 = hash13(i + vec3(0, 1, 1));
    float n111 = hash13(i + vec3(1, 1, 1));

    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z);
}

// Fractal sum of value noise, normalized so that it stays in [0,1] with a mean of
// 0.5 whatever the number of octaves is. The octaves are offset from each other
// as well as scaled, so that they do not line up on the same lattice.
//
// `width` is the distance between the samples of the march that reads this, in the
// units p is in: an octave whose wavelength is finer than that cannot be resolved
// by the march at all, and sampling it anyway is what leaves the grain a cloud
// shows -- an error that differs between neighbouring rays (a weave over the layer)
// and, for one ray, between the frames (a shimmer of it). What cannot be resolved
// is therefore faded out, with a smooth hand-over, so that the finest detail a
// cloud keeps is the finest the march that crosses it can carry.
float cloudNoise(vec3 p, int octaves, float width)
{
    float v = 0.0;
    float amp = 0.5;
    float norm = 0.0;

    for (int i = 0; i < octaves; i++)
    {
        // The octave's wavelength in the units of p is 1/2^i, the argument this
        // sums being doubled with every octave.
        float wavelength = exp2(-float(i));
        float fade = smoothstep(width, width * 2.5, wavelength);

        v += amp * fade * valueNoise3(p);
        norm += amp * fade;
        p = p * 2.03 + vec3(1.7, 9.2, 3.9);
        amp *= 0.5;
    }

    // Normalizing by what is left of the sum keeps the mean at 0.5, so that the
    // coverage threshold still reads as the fraction of the sky left clear.
    return v / max(norm, 1.0e-4);
}

// Shape of the cloud in [0,1]: 0 is clear air, 1 is the densest core. The fBm is
// pushed through the CDF of the normal distribution it is spread like, which
// spends its whole range instead of leaving it bunched around its mean -- the
// coverage threshold can then be read as the fraction of the sky it lets through.
float cloudShape(vec3 p, int octaves, float width)
{
    float n = cloudNoise(p, octaves, width);
    float z = (n - CLOUD_NOISE_MEAN) / CLOUD_NOISE_SIGMA;
    return clamp(0.5 + 0.5 * z / sqrt(1.0 + z * z), 0.0, 1.0);
}

// --- the cloud layer ---------------------------------------------------------

struct CloudLayer
{
    float altitude;     // height of the bottom of the layer over the eye
    float thickness;    // depth of the layer
    float coverage;     // how much of the shape is thrown away, in [0,1)
    float density;      // opacity of a cloud of full shape
    float detail;       // how much the fine noise eats into the shape
    float time;         // seconds, for the drift
    float speed;        // drift speed the host asks for
};

// A cloud is a slab with a flat base and a rounded top: the layer ends towards
// both, or it would be a box with a lid on it.
float cloudHeightProfile(float h)
{
    const float base = 0.12;
    const float top = 0.55;
    return smoothstep(0.0, base, h) * (1.0 - smoothstep(top, 1.0, h));
}

// Density of the cloud at a point, in [0,1]: 0 is clear air. `p` is a point of the
// world (z up), its height measured from the eye. `width` is how far apart the
// samples of the march that asks are, in world units: the noise it cannot resolve
// is faded out rather than sampled as grain (cloudNoise).
float cloudDensity(CloudLayer layer, vec3 p, bool detail, float width)
{
    float h = (p.z - layer.altitude) / layer.thickness;
    if (h <= 0.0 || h >= 1.0)
    {
        return 0.0;
    }

    float profile = cloudHeightProfile(h);
    if (profile <= 0.0)
    {
        return 0.0;
    }

    // the layer drifts as a whole; the fine noise rides a little faster on it
    vec2 wind = vec2(layer.time * layer.speed * 30.0, layer.time * layer.speed * 12.0);
    float frequency = CLOUD_FREQUENCY / layer.thickness;

    float shape = cloudShape(vec3(p.xy + wind, p.z) * frequency, CLOUD_OCTAVES, width * frequency);

    float d = (shape - layer.coverage) / max(1.0 - layer.coverage, 1.0e-3);
    d = pow(clamp(d, 0.0, 1.0), CLOUD_SHAPE_POWER);
    if (d <= 0.0)
    {
        return 0.0;
    }

    if (detail)
    {
        // The fine noise is subtracted from the shape rather than multiplied into
        // it, which is what takes the cloud's edge apart and gives its silhouette
        // the ragged look of a real one.
        vec3 q = vec3(p.xy + wind * 1.7, p.z * 0.75) * frequency * CLOUD_DETAIL_FREQUENCY;
        float erosion = layer.detail * cloudShape(q, CLOUD_DETAIL_OCTAVES, width * frequency * CLOUD_DETAIL_FREQUENCY);
        d = clamp((d - erosion) / max(1.0 - erosion, 1.0e-3), 0.0, 1.0);
    }

    return d * profile;
}

// Optical depth of a column of cloud whose density adds up to `column`, as
// cloudDensity() sums it along the way the light came: the extinction of a cloud
// of full shape, spread over the depth of the layer, is what turns a density into
// an amount of light that survives it.
float cloudOpticalDepth(CloudLayer layer, float column)
{
    return column * layer.density * CLOUD_EXTINCTION / layer.thickness;
}

// Optical depth of the cloud between a point in the layer and the sun, in the
// same units cloudDensity() returns.
//
// The march goes towards the sun through a cone that widens with the distance
// travelled, so that what is being asked is how much of the sky AROUND the sun is
// clouded, which is what a real cloud sees (cloudSunSample). The fine noise is
// left out of it: it is what the widening of the light would blur anyway, and this
// is the most called function of the pass.
//
// It ends at the top of the layer and goes no further: above the layer there is
// nothing but sky, and a sample standing high in the layer has little left to
// cross at all -- the space above the cloud is what this march skips for free,
// having no structure of the layer to consult for the rest of it.
float cloudSunDepth(CloudLayer layer, vec3 p, vec3 sunDir, int steps)
{
    float span = max((layer.altitude + layer.thickness - p.z) / max(sunDir.z, 1.0e-3), 0.0);
    if (span <= 0.0)
    {
        return 0.0;
    }

    float dt = span / float(steps);
    float depth = 0.0;

    for (int i = 0; i < steps; i++)
    {
        // The step is sampled in the middle of the stretch it stands for, so that a
        // thin cloud is not lost to the base of the layer falling between two of them.
        depth += cloudDensity(layer, cloudSunSample(p, sunDir, (float(i) + 0.5) * dt, dt), false, dt) * dt;
    }

    return depth;
}
