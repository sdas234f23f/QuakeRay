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

// How much taller the noise is read than it is wide, over the depth of the layer:
// the height of a point is divided by this before the noise is looked up, so that
// a cloud stands twice as tall as it is broad rather than as flat as the slab it
// sits in. What that is for: the noise is scaled by the thickness of the layer, so
// with this at 1 a layer of 900 units carries shape octaves of 450, 225 and 112
// units along every axis, which is two to eight floors of cloud across the depth of
// it -- each floor a flat mass of its own, and the parallax of a moving eye slides
// one over another, so that a cloud is read as a stack of separate layers. Stood on
// end, the same noise is a body a couple of towers deep, which is what a sky looked
// at from below is made of.
const float CLOUD_VERTICAL_STRETCH = 2.0;

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

// The most steps a walk of a column towards the sun may take. The sky falls back on
// a walk of the same shape as the one the volume of the layer's shadow is filled
// with (cloudSunDepth), so the bound is the one that pass holds itself to.
const int CLOUD_SUN_STEPS_MAX = 64;

// The steps the walk of a column of cloud towards the sun is made of, from the base
// of the layer to its top. One such walk fills the volume of the layer's shadow
// (CmCloudShadow.comp), slice by slice, and the sky falls back on the very same walk
// where that volume does not reach (cloudSunDepth): the two have to be the same
// number for the same column, or the edge of the volume stands in the light of the
// clouds as a square over the sky, so the count lives here rather than in a pass or
// in the host.
//
// Thirty-two of them. What the count resolves is the noise along the column: the
// shape octaves are 900, 450 and 225 world units long over the depth of the layer
// (they are read with the height stretched, CLOUD_VERTICAL_STRETCH), and a step is
// the layer's depth over the height of the sun over this count -- 49 units at a sun
// forty degrees up. Measured against a walk of the same column 512 steps deep, the
// light of a height between two of the volume's slices is off by 0.065 of tau at
// sixteen steps and by 0.020 at thirty-two, which is the difference between a
// ripple of about six per cent of the light of the layer, read as a layering of the
// clouds, and one of two per cent. The walk is the dearest thing the layer does, but
// what it costs is paid where the sky has no volume to read -- the horizon and the
// samples beyond the window -- and by the fill of a few million texels, so the finer
// walk is bought with a fraction of a millisecond rather than with memory.
const int CLOUD_SHADOW_STEPS = 32;

// The steps of the sequences the taps of a march are spread by, taken from the
// golden ratio: successive taps land in different parts of the stretch they stand
// for (a low-discrepancy sequence), rather than anywhere at all (white noise),
// which leaves a smaller and a finer error behind.
const float CLOUD_SEQUENCE_STEP = 0.6180339887; // along the march

// How coarsely the world is divided by the seed that turns the sequence of a march
// (CmSkyClouds.comp): a distance far larger than a texel of the sky's map, so that
// the texels around one column draw nearly the same seed and the error the sequence
// leaves behind is a smooth field of the sky rather than a grain of the map. A
// grain is what the copying of the map's quarters cannot carry (it reads bilinearly
// and so averages the grain of the neighbours), and it is what shivers.
const float CLOUD_SEED_SCALE = 256.0;

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
float cloudNoise(vec3 p, int octaves)
{
    float v = 0.0;
    float amp = 0.5;
    float norm = 0.0;

    for (int i = 0; i < octaves; i++)
    {
        v += amp * valueNoise3(p);
        norm += amp;
        p = p * 2.03 + vec3(1.7, 9.2, 3.9);
        amp *= 0.5;
    }

    return v / norm;
}

// Shape of the cloud in [0,1]: 0 is clear air, 1 is the densest core. The fBm is
// pushed through the CDF of the normal distribution it is spread like, which
// spends its whole range instead of leaving it bunched around its mean -- the
// coverage threshold can then be read as the fraction of the sky it lets through.
float cloudShape(vec3 p, int octaves)
{
    float n = cloudNoise(p, octaves);
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
// world (z up), its height measured from the eye.
float cloudDensity(CloudLayer layer, vec3 p, bool detail)
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

    float shape = cloudShape(vec3(p.xy + wind, p.z / CLOUD_VERTICAL_STRETCH) * frequency, CLOUD_OCTAVES);

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
        //
        // It rides the same wind as the shape does, and it has to: the copy of a
        // texel in the layer's map (CmSkyClouds.comp) is the march of the frame
        // before, put where the column of cloud it holds has moved to, and that
        // move is measured with the shape's own drift (cloudAnchorDelta, from the
        // host). A fine noise advected at another speed than the one the copies are
        // tracked with leaves every edge of every cloud with a standing difference
        // between the texels that were marched this frame and the ones that were
        // copied -- a difference that follows the wind and that no dither of the
        // ages can be asked to hide.
        vec3 q = vec3(p.xy + wind, p.z * 0.75 / CLOUD_VERTICAL_STRETCH) * frequency * CLOUD_DETAIL_FREQUENCY;
        float erosion = layer.detail * cloudShape(q, CLOUD_DETAIL_OCTAVES);
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
// This is the walk the volume of the layer's shadow is filled with -- the column
// over the base point of p, from the base of the layer to its top, in the steps the
// pass that fills it marches (CmCloudShadow.comp) and through the same cone around
// the sun (cloudSunSample) -- and the reading of that volume's slices is repeated
// at the end (CloudShadowMap.h reads them the same way). `steps` and `slices` are
// the ones that volume was made with, and they have to be its own.
//
// Why such a walk rather than one of its own from p: this is what the sky falls
// back on where the volume does not reach (CmSkyClouds.comp), and the two answers
// meet at the edge of the volume. A march of its own, however similar, is a
// different number for the same column -- a coarser step, another phase of the cone,
// no interpolation of the slices -- and the difference stands in the light of the
// clouds as a square over the sky. Being the very number the volume holds, this
// turns the edge into nothing at all.
//
// The fine noise is left out of it: it is what the widening of the light would blur
// anyway, and this is the most called function of the pass. The cone is what asks
// how much of the sky AROUND the sun is clouded, which is what a real cloud sees.
float cloudSunDepth(CloudLayer layer, vec3 p, vec3 sunDir, int steps, int slices)
{
    steps = clamp(steps, 1, CLOUD_SUN_STEPS_MAX);
    slices = max(slices, 2);

    float height = clamp((p.z - layer.altitude) / layer.thickness, 0.0, 1.0);

    // The column stand where the sun ray through p crosses the base of the layer,
    // which is where the volume's own walk of it stood: a ray is a straight line and
    // the same column of cloud is what every point on it sees.
    float dt = layer.thickness / max(sunDir.z, 1.0e-3) / float(steps);
    vec3 base = p + sunDir * ((layer.altitude - p.z) / max(sunDir.z, 1.0e-3));

    // The two slices the height of p stands between, and how far it is through them:
    // the sampler that reads the volume interpolates between the same two, by the
    // same fraction.
    float slice = height * float(slices - 1);
    float low = floor(slice) / float(slices - 1);
    float high = min(low + 1.0 / float(slices - 1), 1.0);
    float through = slice - floor(slice);

    // One traversal of the column answers for both slices: what a slice holds is
    // the part of each step standing above its own height (CmCloudShadow.comp).
    float columnLow = 0.0;
    float columnHigh = 0.0;

    for (int i = 0; i < steps; i++)
    {
        // The step is sampled in the middle of the stretch it stands for, so that a
        // thin cloud is not lost to the base of the layer falling between two of them.
        float column = cloudDensity(layer, cloudSunSample(base, sunDir, (float(i) + 0.5) * dt, dt), false) * dt;
        float step = float(i + 1);
        columnLow  += column * clamp(step - low * float(steps), 0.0, 1.0);
        columnHigh += column * clamp(step - high * float(steps), 0.0, 1.0);
    }

    return cloudOpticalDepth(layer, mix(columnLow, columnHigh, through));
}
